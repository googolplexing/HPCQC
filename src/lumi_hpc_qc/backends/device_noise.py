# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Independent, clean-room implementation. The general approach (depolarizing
# control error plus thermal relaxation, parameterised by operation duration)
# follows publicly documented behaviour of Qiskit Aer's noise tooling and
# IQM's Apache-2.0 fake backends, but NO source code from those projects was
# copied or adapted. All code here is original HPCQC code.
"""Device-calibrated noise model -- the "device-calibrated" noise source.

Use this on a circuit that has already been transpiled to the device's
native gates (PRX -> Qiskit "r", CZ -> "cz"), routed onto the device's
qubit layout, and scheduled (so idle time shows up as explicit "delay"
steps).

How the noise is split:

  build_control_readout_noise_model() -- the part that happens DURING an
  operation:
    - gate control error (the gate is slightly imperfect), from
      single_gate_error / cz_error
    - readout error (the measurement misreads), from readout_fidelity

  build_relaxation_pass() -- the part that happens over TIME:
    - T1/T2 decoherence on every gate and every idle wait, scaled by how
      long that step actually takes. A 20 ns gate decoheres a little; a
      1576 ns readout or a long idle wait decoheres more.

Why split it this way: on real hardware a qubit decoheres for however long
it sits there, whether it's running a gate or just waiting. So all the
time-based decoherence goes through one duration-aware pass, and only the
gate's control error is kept separate. This is closer to a real QPU than
IQM's fake backends, which never model decoherence on idle qubits.

Known limitations:
  - Standard T1/T2 model (simple exponential decay); doesn't capture more
    exotic noise.
  - Gate durations (PRX 20 ns / CZ 60 ns / readout 1576 ns) are typical VTT
    Q50 values, not measured per-calibration -- so the decoherence amounts
    carry that uncertainty. T1/T2 and the gate errors themselves are live
    calibration values.

t2_mode picks which T2 to use: "ramsey" (t2_us, the default -- correct for
idle qubits that aren't being actively protected) or "echo" (t2_echo_us).
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
    """Cap T2 at 2*T1, which is the physical maximum.

    T1 is the energy-relaxation time (how fast the excited state decays to the
    ground state). T2 is the dephasing time (how fast a superposition loses
    its phase relationship). The two are linked by

        1/T2 = 1/(2*T1) + 1/T_phi

    where T_phi is "pure" dephasing from phase noise alone. Energy relaxation
    itself destroys phase as a side effect, contributing the 1/(2*T1) term, so
    even with no pure dephasing at all (T_phi -> infinity) the best possible
    case is T2 = 2*T1. T2 can never exceed that.

    Real calibration occasionally reports a T2 slightly above 2*T1 due to
    measurement scatter, and Qiskit Aer rejects such values as unphysical.
    Capping at 2*T1 keeps the channel valid while only nudging values that
    were already at the boundary.
    """
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
    """Build the noise that happens during gates and measurement.

    Covers gate control error (gates are imperfect) and readout error
    (measurements misread). Does NOT cover T1/T2 decoherence -- that is
    handled separately by build_relaxation_pass, because decoherence depends
    on how long each step takes.

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

    # --- Single-qubit gate control error ---
    # When a single-qubit gate runs, the applied rotation is slightly off from
    # the intended one. We model that imperfection as a depolarizing error
    # whose strength is the gate's measured error rate (1 - RB fidelity,
    # stored as single_gate_error). It is attached to every native
    # single-qubit gate name on that qubit. This is ONLY the control
    # imperfection; the qubit's decoherence during the gate is added later by
    # the relaxation pass.
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

    # --- Two-qubit (CZ) gate control error ---
    # Same idea for the two-qubit CZ gate: a depolarizing error sized by the
    # measured CZ error rate (cz_error), attached to each calibrated qubit
    # pair. We add it in both qubit orderings ([i,j] and [j,i]) because the
    # transpiler may emit the CZ in either direction.
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

    # --- Readout error ---
    # The measurement sometimes reports the wrong bit. We model it as a
    # symmetric bit-flip: probability p of reading 1 when the qubit is 0 and
    # vice versa, where p is derived from the qubit's readout fidelity. (We
    # split the total infidelity evenly between the two flip directions.)
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
    dt_seconds: float | None = None,
):
    """Build the time-based decoherence (T1/T2) as a duration-aware pass.

    This is the part of the noise that depends on HOW LONG each step takes.
    Qiskit Aer's RelaxationNoisePass walks through an already-scheduled
    circuit, reads the real duration of each step, and applies the matching
    amount of T1/T2 decay. A short 20 ns gate gets a little decay; a long
    idle wait (a "delay" step the scheduler inserted while this qubit waited
    for its neighbours) gets proportionally more. This is what lets idle
    qubits decohere realistically -- the feature the IQM fake backends lack.

    Why a pass and not the static noise model: the static noise model applies
    a fixed channel to a given operation regardless of how long it lasts, so
    it cannot tell a 60 ns wait from a 1500 ns wait. The pass can, because it
    inspects each step's scheduled duration. That is the whole reason the
    time-based decoherence lives here instead of in the noise model.

    Which steps get decoherence (op_types): gates and idle delays, but NOT
    measurement. Two reasons: (1) the static noise model already handles the
    measurement outcome via readout error, and (2) attaching a relaxation
    channel to the measure step can interfere with how Aer samples the
    measurement. (Modelling the qubit decohering *during* the long readout
    window is a possible future refinement, deliberately left out here.)

    Units. T1 and T2 are passed in SECONDS (the calibration stores them in
    microseconds, so we multiply by 1e-6). Separately, when the runner
    schedules the circuit it expresses every duration as an integer count of
    "dt" ticks, and it sets one tick = 1 nanosecond (dt = 1e-9 s). Passing
    dt_seconds = 1e-9 here tells the pass to convert a step lasting N ticks
    into N * 1e-9 seconds before comparing it to T1/T2 in seconds. So a CZ
    scheduled as 60 ticks becomes 60 ns of real decay time. Both sides
    (T1/T2 and the durations) end up in seconds, which is what makes the
    decay amounts come out physically correct.

    Returns:
        (relaxation_pass, t1s_s, t2s_s) -- the pass plus the per-qubit T1/T2
        in seconds, indexed in the selected-qubit order, for logging.
    """
    from qiskit_aer.noise import RelaxationNoisePass
    from qiskit.circuit import Gate, Delay

    t2k = _t2_key(t2_mode)
    cal = _load_calibration(calibration_path)
    selected = _select_qubits(cal, num_qubits)

    # Calibration stores T1/T2 in microseconds; the pass wants seconds.
    t1s_s = []
    t2s_s = []
    for _, qdata in selected:
        t1 = qdata.get("t1_us", 50.0) * 1e-6
        t2 = _clamp_t2(t1, qdata.get(t2k, 20.0) * 1e-6)
        t1s_s.append(t1)
        t2s_s.append(t2)

    # Apply decoherence to gates and idle delays only (not measurement).
    op_types = [Gate, Delay]
    kwargs = dict(t1s=t1s_s, t2s=t2s_s, op_types=op_types)
    if dt_seconds is not None:
        kwargs["dt"] = dt_seconds
    relax_pass = RelaxationNoisePass(**kwargs)
    return relax_pass, t1s_s, t2s_s

