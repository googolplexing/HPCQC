# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Independent, clean-room implementation. The general approach (depolarizing
# control error plus thermal relaxation, parameterised by operation duration)
# follows publicly documented behaviour of Qiskit Aer's noise tooling and
# IQM's Apache-2.0 fake backends, but NO source code from those projects was
# copied or adapted. All code here is original HPCQC code.
"""Device-calibrated noise for HPCQC -- the ``device-calibrated`` noise source.

For circuits transpiled to a device's native gates (PRX -> Qiskit ``r``,
CZ -> ``cz``), routed onto the device coupling map, and SCHEDULED (ALAP +
PadDelay) so idle periods are explicit ``delay`` instructions.

PHYSICS MODEL (Option 1 -- uniform duration-driven decoherence)
---------------------------------------------------------------
On real hardware, T1/T2 decoherence is one continuous process: a qubit
relaxes/dephases for whatever wall-clock time elapses, identically whether
that time is spent executing a gate or sitting idle. The only thing unique
to a gate is the coherent CONTROL error (imperfect rotation) on top.

So decoherence is modelled as a SINGLE duration-aware mechanism applied to
every instruction by its real scheduled length, and the gate control error
is layered separately:

  * Static ``NoiseModel`` (this module, build_control_readout_noise_model):
      - depolarizing CONTROL error on native gates, from single_gate_error /
        cz_error (1 - RB fidelity);
      - symmetric readout error, from readout_fidelity.
    NO thermal relaxation lives here.

  * ``RelaxationNoisePass`` (this module, build_relaxation_pass):
      - ALL T1/T2 decoherence, on EVERY instruction (gates AND delays),
        scaled by each instruction's actual scheduled duration. This is the
        only mechanism that reads real durations, so a 20 ns PRX, a 60 ns CZ,
        a 1576 ns readout, and a variable-length idle delay each decohere by
        the correct, different amount.

This is more faithful to a real QPU than IQM's fake backends, which apply
relaxation only during gates (fixed nominal duration) and never to idle
qubits. The idle-decoherence term -- a qubit waiting through its neighbours'
gates and routing SWAPs -- is captured here and is absent there.

HONEST LIMITS
-------------
  * Still a Markovian T1/T2 model (memoryless exponential decay at the
    calibrated constants). Real non-Markovian / time-correlated noise is not
    captured -- this is the standard framework, not a transcendence of it.
  * Gate durations are representative VTT Q50 demonstrator-sheet values
    (PRX 20 / CZ 60 / readout 1576 ns), not per-calibration-measured, so the
    relaxation AMOUNTS carry that duration uncertainty. The control errors
    and T1/T2 themselves are live calibration values.

T2 source: t2_mode='ramsey' (t2_us; default, correct for un-refocused idle
qubits) or 'echo' (t2_echo_us; only under dynamical decoupling).
"""
from __future__ import annotations

from lumi_hpc_qc.backends.noise_model import (
    _select_qubits,
    _load_calibration,
    _extract_edges,
)

_NATIVE_1Q_GATES = ["r", "rz", "sx", "x"]
_NATIVE_2Q_GATES = ["cz"]
_DEFAULT_T2_MODE = "ramsey"

_T2_WARN_US = 1.0
_CZ_ERR_WARN = 0.05


def _t2_key(t2_mode: str) -> str:
    if t2_mode not in ("ramsey", "echo"):
        raise ValueError(f"t2_mode must be 'ramsey' or 'echo', got {t2_mode!r}")
    return "t2_us" if t2_mode == "ramsey" else "t2_echo_us"


def _clamp_t2(t1_ns: float, t2_ns: float) -> float:
    """Aer requires T2 <= 2*T1 for a physical relaxation channel."""
    return min(t2_ns, 2.0 * t1_ns)


def _resolve_durations(cal, sg, cz, me):
    sg = sg if sg is not None else cal.get("single_gate_time_ns", 20)
    cz = cz if cz is not None else cal.get("cz_gate_time_ns", 60)
    me = me if me is not None else cal.get("measure_time_ns", 1576)
    return float(sg), float(cz), float(me)


def build_control_readout_noise_model(
    calibration_path: str,
    num_qubits: int = 10,
    t2_mode: str = _DEFAULT_T2_MODE,
    single_gate_time_ns: float | None = None,
    cz_gate_time_ns: float | None = None,
    measure_time_ns: float | None = None,
):
    """Static NoiseModel: depolarizing CONTROL error on gates + readout only.

    NO thermal relaxation here -- all decoherence is handled by the
    RelaxationNoisePass (see build_relaxation_pass), per the Option-1 physics
    model documented at module level.

    Returns:
        (noise_model, coupling_map, info)
    """
    from qiskit_aer.noise import NoiseModel, ReadoutError
    from qiskit_aer.noise.errors import depolarizing_error
    from qiskit.transpiler import CouplingMap

    t2k = _t2_key(t2_mode)
    cal = _load_calibration(calibration_path)
    selected = _select_qubits(cal, num_qubits)
    name_to_idx = {qname: i for i, (qname, _) in enumerate(selected)}
    gates_data = cal.get("two_qubit_gates", {})
    sg_ns, cz_ns, me_ns = _resolve_durations(
        cal, single_gate_time_ns, cz_gate_time_ns, measure_time_ns
    )

    noise_model = NoiseModel()
    warnings: list[str] = []

    # Single-qubit CONTROL error (depolarizing) on native 1q gates.
    for qname, qdata in selected:
        i = name_to_idx[qname]
        sg_err = qdata.get("single_gate_error", 0.001)
        if qdata.get(t2k, 99.0) < _T2_WARN_US:
            warnings.append(
                f"qubit {qname}: {t2_mode} T2={qdata.get(t2k):.3f}us "
                f"(<{_T2_WARN_US}us) -- heavy dephasing"
            )
        if sg_err > 0:
            noise_model.add_quantum_error(
                depolarizing_error(sg_err, 1), _NATIVE_1Q_GATES, [i]
            )

    # Two-qubit CONTROL error (depolarizing) on native cz.
    for gate_pair, gate_data in gates_data.items():
        parts = gate_pair.split("-")
        if len(parts) != 2:
            continue
        q1, q2 = parts
        if q1 not in name_to_idx or q2 not in name_to_idx:
            continue
        i, j = name_to_idx[q1], name_to_idx[q2]
        cz_err = gate_data.get("cz_error", 0.005)
        if cz_err > _CZ_ERR_WARN:
            warnings.append(
                f"edge {gate_pair}: CZ error={cz_err:.4f} "
                f"(>{_CZ_ERR_WARN}) -- low-fidelity coupling"
            )
        if cz_err > 0:
            err2 = depolarizing_error(cz_err, 2)
            noise_model.add_quantum_error(err2, _NATIVE_2Q_GATES, [i, j])
            noise_model.add_quantum_error(err2, _NATIVE_2Q_GATES, [j, i])

    # Readout (symmetric).
    for qname, qdata in selected:
        i = name_to_idx[qname]
        ro = qdata.get("readout_fidelity", 0.97)
        p = (1 - ro) / 2
        noise_model.add_readout_error(
            ReadoutError([[1 - p, p], [p, 1 - p]]), [i]
        )

    edges = _extract_edges(cal, name_to_idx)
    coupling_map = CouplingMap(edges) if edges else None

    info = {
        "selected_qubits": [q[0] for q in selected],
        "t2_mode": t2_mode,
        "single_gate_time_ns": sg_ns,
        "cz_gate_time_ns": cz_ns,
        "measure_time_ns": me_ns,
        "duration_source": cal.get("duration_source", "unknown"),
        "num_edges": (len(edges) // 2) if edges else 0,
        "health_warnings": warnings,
    }
    return noise_model, coupling_map, info


def build_relaxation_pass(
    calibration_path: str,
    num_qubits: int = 10,
    t2_mode: str = _DEFAULT_T2_MODE,
    dt_seconds: float = 1e-9,
):
    """ALL T1/T2 decoherence as a duration-aware RelaxationNoisePass.

    Applied by the runner to the SCHEDULED circuit. Adds duration-dependent
    thermal relaxation after EVERY instruction (op_types=None), so gates
    decohere by their gate duration and idle delays by their wait duration --
    one uniform physical process across the whole timeline.

    T1/T2 arrays are indexed in the SELECTED-qubit order, matching the noise
    model and coupling map from build_control_readout_noise_model.

    dt_seconds: sample-time used to convert scheduled dt-unit durations to
    seconds. The runner sets InstructionDurations in 'dt' with dt=1 ns so a
    duration of N dt == N ns; with T1/T2 supplied in ns and dt=1e-9 s, the
    relaxation math is consistent. (T1/T2 are passed in ns and dt in s; Aer
    multiplies the scheduled dt-count by dt to get seconds, then compares to
    T1/T2 -- so we pass T1/T2 in seconds too. See below.)

    Returns:
        (relaxation_pass, t1s_s, t2s_s)  -- pass plus per-index T1/T2 in
        SECONDS (for logging / provenance).
    """
    from qiskit_aer.noise import RelaxationNoisePass

    t2k = _t2_key(t2_mode)
    cal = _load_calibration(calibration_path)
    selected = _select_qubits(cal, num_qubits)

    # Work in SECONDS for the pass: T1/T2 us -> s; durations are scheduled in
    # dt units and the runner sets dt = 1e-9 s, so a duration of N (dt) means
    # N nanoseconds = N * dt seconds. RelaxationNoisePass(t1s, t2s, dt) then
    # computes relaxation over (scheduled_dt_count * dt) seconds against t1s/
    # t2s in seconds -- fully consistent.
    t1s_s = []
    t2s_s = []
    for _, qdata in selected:
        t1 = qdata.get("t1_us", 50.0) * 1e-6           # us -> s
        t2 = _clamp_t2(t1, qdata.get(t2k, 20.0) * 1e-6)  # us -> s, clamped
        t1s_s.append(t1)
        t2s_s.append(t2)

    relax_pass = RelaxationNoisePass(t1s=t1s_s, t2s=t2s_s, dt=dt_seconds)
    return relax_pass, t1s_s, t2s_s
