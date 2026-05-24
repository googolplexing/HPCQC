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
_NATIVE_BASIS = ["r", "rz", "sx", "x", "cz", "id", "measure"]

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
    """Depth / op-count comparison for the deepest circuit in the batch.

    Reports the LOGICAL (input) circuit against its NATIVE (decomposed + routed
    + ALAP-scheduled) form, so the gate inflation and scheduling impact of
    native compilation are visible. The "deepest" circuit (largest logical
    depth) is chosen because it is the dominant-cost / worst-case member of the
    batch; transpile() preserves input order, so the native circuit at the same
    index corresponds to it.

    2q-depth (depth counting only two-qubit gates) is the most robust
    complexity measure here: it is unaffected by how 1q gates and idle Delays
    are laid out, and 2q gates dominate both runtime and infidelity.

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
        m = {
            "num_circuits": len(logical_circuits),
            "deepest_index": idx,
            "logical_depth": lc.depth(),
            "native_depth": nc.depth(),
            "logical_2q_depth": lc.depth(_is_2q),
            "native_2q_depth": nc.depth(_is_2q),
            "logical_2q_count": sum(1 for i in lc.data if _is_2q(i)),
            "native_2q_count": sum(1 for i in nc.data if _is_2q(i)),
            "native_ops": dict(nc.count_ops()),
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
                               verbose=True):
    # Imported here (not at module top) so callers that only need the noiseless
    # or iqm paths don't pay for the transpiler-internals import.
    from qiskit.transpiler import InstructionDurations, CouplingMap, Target

    sg_ns, cz_ns, me_ns = durations
    nm, coupling_map, info = build_control_readout_noise_model(
        calibration_path, num_qubits=num_qubits, t2_mode=t2_mode,
        single_gate_time_ns=sg_ns, cz_gate_time_ns=cz_ns, measure_time_ns=me_ns,
        spec=spec,
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
    scheduled = transpile(
        circuits,
        target=target,
        scheduling_method="alap",
        optimization_level=optimization_level,
        num_processes=num_processes,
    )

    # Circuit-complexity metrics: logical (input) vs native (decomposed + routed
    # + scheduled). Always stored in info for structured consumption; printed
    # here only when verbose. This is what quantifies the gate inflation of
    # native compilation (the logical rzz chain expanding into CZ + single-qubit
    # gates, plus routing SWAPs and scheduled Delays).
    metrics = _safe_circuit_metrics(circuits, scheduled)
    info["circuit_metrics"] = metrics
    if verbose and metrics:
        print(
            f"[prepare] device-calibrated circuit metrics "
            f"(deepest of {metrics['num_circuits']}, idx {metrics['deepest_index']}): "
            f"depth {metrics['logical_depth']} -> {metrics['native_depth']} "
            f"(x{metrics['depth_ratio']}); "
            f"2q-depth {metrics['logical_2q_depth']} -> {metrics['native_2q_depth']}; "
            f"2q gates {metrics['logical_2q_count']} -> {metrics['native_2q_count']}; "
            f"native ops {metrics['native_ops']}"
        )

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
