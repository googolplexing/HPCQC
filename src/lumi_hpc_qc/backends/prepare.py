# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Simulation preparation seam -- one entry point for the three noise sources.

Any HPCQC experiment that wants to run a batch of circuits under a chosen
fidelity model calls ONE function here:

    run_circuits, simulator, prep = prepare_simulation(
        circuits, source="device-calibrated", spec=noise_spec,
        calibration_path=cal, num_qubits=10, durations=(None, None, None))
    result = simulator.run(run_circuits, shots=1000, memory=True).result()

`source` is one of:
  "noiseless"         -- AerSimulator(), no noise; circuit transpiled as-written.
  "device-calibrated" -- native-gate decomposition + routing + ALAP scheduling +
                         the live-calibration noise model (control + readout +
                         gate-duration relaxation resident, idle/delay relaxation
                         as a custom pass), run under method="statevector". The
                         only mode with real idle decoherence. `spec` (a
                         lumi_hpc_qc.backends.noise_spec.NoiseSpec) selects which
                         channels are active; None => full model.
  "iqm-fake-backend"  -- IQM's local FakeBackend (static baked noise, no idle
                         relaxation); requires the `iqm` package in the runtime.

This is deliberately the SAME preparation that floquet_runner uses, extracted so
the Phase E sweep engine (and any future experiment) shares one validated code
path instead of re-deriving native transpilation / scheduling / the statevector
Kraus-safety pin. The empty-Kraus reasoning lives entirely in the
device-calibrated branch (the method="statevector" pin); callers do not need to
know about it.

Returns a PreparedSimulation dataclass; unpacking the first three fields
(run_circuits, simulator, prep) is supported for convenience.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qiskit import transpile
from qiskit_aer import AerSimulator

from lumi_hpc_qc.backends.device_noise import (
    build_control_readout_noise_model,
    build_relaxation_pass,
)

# Native basis for the device-calibrated path. rz is a virtual frame change
# (zero duration on real hardware); r is PRX; cz is the native entangler.
_NATIVE_BASIS = ["r", "rz", "sx", "x", "cz", "id", "measure", "delay"]
# "delay" is included so Target.from_configuration registers Delay as a
# supported variable-width 1-qubit instruction. Without it,
# PadDelay.__delay_supported(qarg) returns False on every qubit and PadDelay
# silently SKIPS inserting idle Delay instructions into the scheduled circuit,
# starving the idle relaxation pass (RelaxationNoisePass with op_types=[Delay])
# of anything to act on. See FINDING-PADDELAY-IDLE-NOT-INSERTED-v1_0.md
# for the full code-path trace and validation evidence.

# Each worker stays single-threaded: many workers share a node, so we do not
# want each spawning its own Aer thread pool.
_AER_SINGLE_THREAD = dict(
    max_parallel_threads=1,
    max_parallel_experiments=1,
    max_parallel_shots=1,
)

VALID_SOURCES = ("noiseless", "device-calibrated", "iqm-fake-backend")


def _is_2q(instr) -> bool:
    op = instr.operation
    return op.num_qubits == 2 and op.name != "barrier"


def _safe_circuit_metrics(logical_circuits, native_circuits) -> dict:
    """Depth / op-count comparison for the batch.

    Reports two complementary views of the LOGICAL (input) circuits vs their
    NATIVE (decomposed + routed + ALAP-scheduled) forms:

      * DEEPEST circuit (largest logical depth): the worst-case single member,
        useful for "how big does one circuit get". transpile() preserves input
        order, so the native circuit at the same index corresponds to it.
      * AGGREGATE over ALL circuits: the real per-instance workload. A Floquet
        instance builds one circuit per kick-count (e.g. 60 circuits: 0..59
        periods), and EVERY one is simulated. So the runtime-relevant quantity
        is the SUM of gates across the whole batch, not the single deepest
        circuit -- the batch is a staircase of growing circuits, and the
        aggregate is the area under it. `total_native_cz` is the headline:
        total CZ Aer must sample noise across, summed over all circuits.

    2q gates / CZ dominate both runtime and infidelity, so they are the focus.

    This is diagnostic only -- it NEVER raises (returns {} on any failure), so a
    metrics quirk can never break an actual simulation run.
    """
    try:
        if not logical_circuits or not native_circuits:
            return {}
        log_depths = [c.depth() for c in logical_circuits]
        idx = max(range(len(log_depths)), key=lambda k: log_depths[k])
        lc = logical_circuits[idx]
        nc = native_circuits[idx] if idx < len(native_circuits) else native_circuits[-1]

        # Single pass over the batch: aggregate totals AND a per-kick row for
        # each circuit. The batch is built in kick order (circuit at index k has
        # k Floquet periods), and transpile() preserves order, so index == kick
        # count and logical_circuits[k] pairs with native_circuits[k].
        total_logical_2q = 0
        total_native_2q = 0
        total_native_cz = 0
        total_native_gates = 0
        per_kick = []
        n = min(len(logical_circuits), len(native_circuits))
        for k in range(n):
            lc_k = logical_circuits[k]
            nc_k = native_circuits[k]
            lc_2q = sum(1 for i in lc_k.data if _is_2q(i))
            ops_k = nc_k.count_ops()
            nc_cz = ops_k.get("cz", 0)
            nc_2q = sum(1 for i in nc_k.data if _is_2q(i))
            nc_tot = sum(ops_k.values())
            total_logical_2q += lc_2q
            total_native_cz += nc_cz
            total_native_2q += nc_2q
            total_native_gates += nc_tot
            # kept compact (one row per kick lands in the result JSON):
            per_kick.append({
                "kick": k,
                "logical_2q": lc_2q,
                "native_cz": nc_cz,
                "native_total": nc_tot,
            })

        m = {
            "num_circuits": len(logical_circuits),
            # --- deepest single circuit (worst case) ---
            "deepest_index": idx,
            "logical_depth": lc.depth(),
            "native_depth": nc.depth(),
            "logical_2q_depth": lc.depth(_is_2q),
            "native_2q_depth": nc.depth(_is_2q),
            "logical_2q_count": sum(1 for i in lc.data if _is_2q(i)),
            "native_2q_count": sum(1 for i in nc.data if _is_2q(i)),
            "native_ops": dict(nc.count_ops()),
            # --- aggregate over ALL circuits (true per-instance workload) ---
            "total_logical_2q": total_logical_2q,
            "total_native_2q": total_native_2q,
            "total_native_cz": total_native_cz,
            "total_native_gates": total_native_gates,
            # --- full per-kick curve (one row per kick) -> result JSON ---
            "per_kick": per_kick,
        }
        ld = m["logical_depth"]
        m["depth_ratio"] = round(m["native_depth"] / ld, 2) if ld else None
        return m
    except Exception:
        return {}


@dataclass
class PreparedSimulation:
    """Everything needed to run a batch of circuits under one fidelity model.

    Attributes:
        run_circuits: Transpiled circuits ready to hand to simulator.run().
        simulator: A configured AerSimulator (noiseless / device-calibrated) or
            an IQM FakeBackend (iqm-fake-backend). Both expose .run(circuits,
            shots=..., memory=...).
        source: The noise source that produced this (echoed back).
        relaxation_active: True if the device-calibrated idle/delay relaxation
            pass was attached (i.e. thermal channel on). False otherwise.
        info: Backend-specific metadata. For device-calibrated this is the dict
            returned by build_control_readout_noise_model (selected_qubits,
            resolved durations, health warnings, ...). Empty for other sources.
    """
    run_circuits: list
    simulator: Any
    source: str
    relaxation_active: bool = False
    info: dict = field(default_factory=dict)

    def __iter__(self):
        # Allow:  run_circuits, simulator, prep = prepare_simulation(...)
        # where `prep` is this object (for relaxation_active / info access).
        yield self.run_circuits
        yield self.simulator
        yield self


def prepare_simulation(
    circuits,
    source: str,
    *,
    spec=None,
    calibration_path: str | None = None,
    num_qubits: int = 10,
    physical_qubits: list[str] | None = None,
    physical_edges: list | None = None,
    durations=(None, None, None),
    t2_mode: str = "ramsey",
    iqm_device: str = "aphrodite",
    optimization_level: int = 3,
    num_processes: int = 1,
    verbose: bool = True,
) -> PreparedSimulation:
    """Transpile `circuits` and build a simulator for the chosen noise source.

    Args:
        circuits: list[QuantumCircuit] -- the logical circuits to prepare.
        source: one of VALID_SOURCES.
        spec: device-calibrated only -- a NoiseSpec selecting active channels
            (None => full model). Ignored by other sources.
        calibration_path: device-calibrated only -- path to the calibration JSON.
        num_qubits: number of qubits to select / route onto.
        physical_qubits: device-calibrated only -- F5a per-placement
            composition. When given (len == num_qubits, names in the
            calibration), the noise model is built from exactly these qubits in
            logical order and the circuit is routed onto them (identity
            initial_layout); logical k -> physical_qubits[k]. None preserves the
            historical fidelity-driven self-selection (and free layout).
        physical_edges: device-calibrated only -- optional placement edges,
            validated as real calibrated two-qubit gates among physical_qubits.
        durations: (single_gate_ns, cz_ns, measure_ns) overrides; any may be
            None to use the calibration/VTT defaults. Device-calibrated only.
        t2_mode: "ramsey" (T2*) or "echo" (Hahn). Device-calibrated only.
        iqm_device: "aphrodite" or "apollo". iqm-fake-backend only.
        optimization_level / num_processes: passed to transpile().
        verbose: device-calibrated only -- if True (default), print a one-line
            circuit-metrics summary (logical vs native depth / 2q gates /
            native op counts). The same metrics are always stored in
            PreparedSimulation.info["circuit_metrics"] regardless, so a
            high-volume caller (e.g. the sweep engine) can pass verbose=False
            and read them structurally instead.

    Returns:
        PreparedSimulation.

    Raises:
        ValueError: unknown source.
        RuntimeError: device-calibrated requested without a calibration path.
        ImportError: iqm-fake-backend requested without the `iqm` package.
    """
    if source not in VALID_SOURCES:
        raise ValueError(
            f"unknown source {source!r}; valid: {', '.join(VALID_SOURCES)}"
        )

    if source == "noiseless":
        return _prepare_noiseless(circuits, optimization_level, num_processes)

    if source == "device-calibrated":
        if not calibration_path:
            raise RuntimeError(
                "device-calibrated requires calibration_path=<JSON>"
            )
        return _prepare_device_calibrated(
            circuits, num_qubits=num_qubits, calibration_path=calibration_path,
            physical_qubits=physical_qubits, physical_edges=physical_edges,
            durations=durations, t2_mode=t2_mode, spec=spec,
            optimization_level=optimization_level, num_processes=num_processes,
            verbose=verbose,
        )

    # iqm-fake-backend
    return _prepare_iqm_fake(
        circuits, iqm_device=iqm_device,
        optimization_level=optimization_level, num_processes=num_processes,
    )


def _prepare_noiseless(circuits, optimization_level, num_processes):
    simulator = AerSimulator(**_AER_SINGLE_THREAD)
    run_circuits = transpile(
        circuits, simulator,
        optimization_level=optimization_level, num_processes=num_processes,
    )
    return PreparedSimulation(
        run_circuits=run_circuits, simulator=simulator, source="noiseless",
        relaxation_active=False, info={},
    )


def _prepare_device_calibrated(circuits, *, num_qubits, calibration_path,
                               durations, t2_mode, spec,
                               optimization_level, num_processes,
                               physical_qubits=None, physical_edges=None,
                               verbose=True):
    # Imported here (not at module top) so callers that only need the noiseless
    # or iqm paths don't pay for the transpiler-internals import.
    from qiskit.transpiler import InstructionDurations, CouplingMap, Target

    sg_ns, cz_ns, me_ns = durations
    nm, coupling_map, info = build_control_readout_noise_model(
        calibration_path, num_qubits=num_qubits, t2_mode=t2_mode,
        single_gate_time_ns=sg_ns, cz_gate_time_ns=cz_ns, measure_time_ns=me_ns,
        spec=spec,
        physical_qubits=physical_qubits, physical_edges=physical_edges,
    )

    # One tick = 1 ns, so a duration value of N means N nanoseconds.
    dt_s = 1e-9
    sg = int(round(info["single_gate_time_ns"]))   # PRX:    ~20 ns
    cz = int(round(info["cz_gate_time_ns"]))        # CZ:     ~60 ns
    me = int(round(info["measure_time_ns"]))        # readout ~1576 ns

    instr_durations = InstructionDurations(
        [
            ("r", None, sg), ("rz", None, 0), ("sx", None, sg),
            ("x", None, sg), ("id", None, sg), ("cz", None, cz),
            ("measure", None, me), ("reset", None, me),
        ],
        dt=dt_s,
    )

    cmap = coupling_map if isinstance(coupling_map, CouplingMap) else (
        CouplingMap(coupling_map) if coupling_map else None)

    # Qiskit 2.3 removed transpile()'s loose instruction_durations kwarg;
    # durations must be carried on a Target. transpile() then does layout,
    # routing, native translation, and ALAP scheduling in one validated
    # pipeline. ALAP scheduling stamps a concrete .duration on EVERY
    # instruction (gates and delays), which the relaxation pass reads to decide
    # how much each one decoheres -- without it, gate relaxation is silently
    # skipped.
    target = Target.from_configuration(
        basis_gates=_NATIVE_BASIS,
        num_qubits=num_qubits,
        coupling_map=cmap,
        instruction_durations=instr_durations,
        dt=dt_s,
    )
    # Runtime precondition: the Target MUST list "delay" as a supported
    # instruction or PadDelay (qiskit.transpiler.passes.scheduling.padding.
    # base_padding.BasePadding.__delay_supported) silently skips Delay
    # insertion on every qubit, starving the RelaxationNoisePass
    # (op_types=[Delay]) of anything to act on. This converts a previously-
    # implicit upstream-library precondition into an in-repo runtime
    # invariant: if a future edit drops "delay" from _NATIVE_BASIS or
    # otherwise fails to register it, the pipeline raises here instead of
    # silently producing noise-deficient results.
    # See FINDING-PADDELAY-IDLE-NOT-INSERTED-v1_0.md.
    if "delay" not in target.operation_names:
        raise RuntimeError(
            "device-calibrated Target does not list \"delay\" as a supported "
            "instruction; PadDelay will silently skip Delay insertion and the "
            "idle relaxation pass (RelaxationNoisePass with op_types=[Delay]) "
            "will be starved. _NATIVE_BASIS must include \"delay\". "
            "See FINDING-PADDELAY-IDLE-NOT-INSERTED-v1_0.md."
        )
    # F5a: when a placement is given, pin logical k -> relabeled index k so the
    # routed circuit's qubit indices line up with the placement-keyed noise
    # model. transpile(initial_layout=None) is exactly today's free-layout
    # behavior, so the no-placement path (incl. the F4 banked reference) is
    # byte-identical.
    initial_layout = list(range(num_qubits)) if physical_qubits is not None else None
    scheduled = transpile(
        circuits,
        target=target,
        scheduling_method="alap",
        optimization_level=optimization_level,
        num_processes=num_processes,
        initial_layout=initial_layout,
    )

    # Circuit-complexity metrics: logical (input) vs native (decomposed + routed
    # + scheduled). Always stored in info for structured consumption; printed
    # here only when verbose. This is what quantifies the gate inflation of
    # native compilation (the logical rzz chain expanding into CZ + single-qubit
    # gates, plus routing SWAPs and scheduled Delays).
    metrics = _safe_circuit_metrics(circuits, scheduled)
    info["circuit_metrics"] = metrics
    if verbose and metrics:
        # Human-readable labeled block. Full structured data (incl. native_ops
        # and every field) is in info["circuit_metrics"] / the result JSON for
        # machine consumption; this block is for a person skimming the log.
        m = metrics
        depth_x = f"{m['depth_ratio']}x deeper" if m['depth_ratio'] else "n/a"
        tot_log2q = m['total_logical_2q']
        tot_cz = m['total_native_cz']
        cz_x = f"{tot_cz / tot_log2q:.1f}x" if tot_log2q else "n/a"
        print(
            f"[prepare] device-calibrated compilation "
            f"({m['num_circuits']} circuits, native gates + ALAP schedule)\n"
            f"  per-instance workload (summed over all {m['num_circuits']} "
            f"circuits -- this is what drives runtime):\n"
            f"      two-qubit gates (CZ) to simulate : {tot_cz:>8,}   "
            f"(was {tot_log2q:,} logical rzz; {cz_x} after rzz->~2 CZ)\n"
            f"      native gates total               : {m['total_native_gates']:>8,}   "
            f"(then multiplied by your shot count)\n"
            f"  largest single circuit (kick {m['deepest_index']} of "
            f"{m['num_circuits'] - 1}):\n"
            f"      circuit depth   : {m['logical_depth']:>5,} logical  ->  "
            f"{m['native_depth']:>5,} native   ({depth_x})\n"
            f"      two-qubit gates : {m['logical_2q_count']:>5,} logical  ->  "
            f"{m['native_2q_count']:>5,} native"
        )
        # 3-row per-kick trend (first / middle / last) -- confirms linear growth
        # at a glance; the full per-kick curve is in the result JSON.
        pk = m.get("per_kick") or []
        if len(pk) >= 2:
            picks = sorted({0, len(pk) // 2, len(pk) - 1})
            print(f"  growth check (CZ per circuit, full curve in JSON "
                  f"under circuit_metrics.per_kick):")
            for j in picks:
                r = pk[j]
                print(f"      kick {r['kick']:>3} : {r['native_cz']:>5,} CZ   "
                      f"({r['native_total']:,} native gates)")

    # Idle/delay relaxation. Gate-time relaxation is already RESIDENT in `nm`;
    # this pass adds the variable-duration idle "delay" relaxation, registered
    # as a NoiseModel custom pass exactly like Aer's NoiseModel.from_backend.
    # Skipped entirely when the thermal channel is off (e.g. --noise=1q,2q).
    thermal_on = (spec is None) or bool(spec.thermal_relaxation)
    relaxation_active = False
    if thermal_on:
        relax_pass, _, _ = build_relaxation_pass(
            calibration_path, num_qubits=num_qubits, t2_mode=t2_mode,
            dt_seconds=dt_s, target=target,
            physical_qubits=physical_qubits,
        )
        nm._custom_noise_passes.append(relax_pass)
        relaxation_active = True

    # Pin method="statevector" rather than "automatic". The thermal channel is a
    # genuine non-unitary Kraus map for any qubit with T2 > T1. Under
    # statevector, Aer samples it via a kraus path that reads a PRECOMPUTED
    # canonical Kraus; that precompute (NoiseModel::enable_kraus_method) runs
    # DETERMINISTICALLY in the forced-method path when method==statevector and
    # the opset contains kraus (aer_controller.hpp:778). In "automatic" mode the
    # precompute is gated on a per-circuit method decision that can be skipped,
    # leaving an empty Kraus -> "QuantumError: Kraus is empty". Forcing
    # statevector closes that gap for EVERY channel combination and is the
    # scalable path (statevector + per-shot sampling, not O(4^n) density_matrix).
    simulator = AerSimulator(
        method="statevector", noise_model=nm, **_AER_SINGLE_THREAD,
    )
    return PreparedSimulation(
        run_circuits=scheduled, simulator=simulator,
        source="device-calibrated", relaxation_active=relaxation_active,
        info=info,
    )


def _prepare_iqm_fake(circuits, *, iqm_device, optimization_level,
                      num_processes):
    # Local IQM FakeBackend: static baked noise, executed locally via Aer; it
    # does NOT contact any VTT/QX endpoint. Requires the `iqm` package, which is
    # not present in every container -- the ImportError is surfaced as-is so the
    # caller can report a clear "No module named 'iqm'".
    from iqm.qiskit_iqm import IQMFakeAphrodite
    try:
        from iqm.qiskit_iqm import IQMFakeApollo
    except Exception:
        IQMFakeApollo = None

    fake_map = {"aphrodite": IQMFakeAphrodite, "apollo": IQMFakeApollo}
    fb_cls = fake_map.get(iqm_device) or IQMFakeAphrodite
    fb = fb_cls()
    run_circuits = transpile(
        circuits, backend=fb,
        optimization_level=optimization_level, num_processes=num_processes,
    )
    return PreparedSimulation(
        run_circuits=run_circuits, simulator=fb, source="iqm-fake-backend",
        relaxation_active=False, info={"fake_backend": fb_cls.__name__},
    )
