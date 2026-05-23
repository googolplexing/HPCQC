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
# Native single-qubit gates with a finite execution time (they decohere) vs the
# virtual rz frame change (zero time -> no relaxation, only control error).
_TIMED_1Q_GATES = ["r", "sx", "x"]
_VIRTUAL_1Q_GATES = ["rz"]
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


def _per_qubit_t1_t2_seconds(cal, selected, t2_mode):
    """Per-qubit (T1, T2) in SECONDS, indexed in the selected-qubit order.

    T2 is the chosen mode (ramsey/echo), capped at the physical 2*T1 ceiling.
    The calibration stores microseconds; thermal_relaxation_error wants the
    same time unit as the gate duration, and we use seconds throughout.
    """
    t2k = _t2_key(t2_mode)
    t1s_s, t2s_s = [], []
    for _, qdata in selected:
        t1 = qdata.get("t1_us", 50.0) * 1e-6
        t2 = _clamp_t2(t1, qdata.get(t2k, 20.0) * 1e-6)
        t1s_s.append(t1)
        t2s_s.append(t2)
    return t1s_s, t2s_s


def _thermal_1q(t1_s, t2_s, dur_s):
    """A single-qubit thermal-relaxation error for a finite-duration step.

    Returns None for a zero (or missing) duration -- a virtual rz takes no
    time and a zero-duration thermal channel would be degenerate.
    """
    if dur_s is None or dur_s <= 0:
        return None
    from qiskit_aer.noise.errors import thermal_relaxation_error
    return thermal_relaxation_error(t1_s, t2_s, dur_s)


def _thermal_nq(qubit_order, t1s_s, t2s_s, dur_s):
    """A multi-qubit thermal-relaxation error: independent single-qubit
    relaxation on each qubit for the same step duration, tensored together in
    the given qubit order.

    The order MUST match the qubit list passed to add_quantum_error so the
    per-qubit channels line up with the gate's operands. Returns None for a
    zero/missing duration.
    """
    if dur_s is None or dur_s <= 0:
        return None
    err = None
    for q in qubit_order:
        single = _thermal_1q(t1s_s[q], t2s_s[q], dur_s)
        err = single if err is None else err.expand(single)
    return err


def _combine_control_relax(depol_err, relax_err):
    """Combine a gate's control (depolarizing) error with its thermal
    relaxation into one QuantumError applied during the gate: control error
    first, then relaxation. Either may be None.
    """
    if depol_err is not None and relax_err is not None:
        return depol_err.compose(relax_err)
    return depol_err if depol_err is not None else relax_err


def build_control_readout_noise_model(
    calibration_path: str,
    num_qubits: int = 10,
    t2_mode: str = _DEFAULT_T2_MODE,
    single_gate_time_ns: float | None = None,
    cz_gate_time_ns: float | None = None,
    measure_time_ns: float | None = None,
):
    """Build the static noise that is applied during gates and measurement.

    Covers, per native gate:
      - control error (the gate is imperfect): a depolarizing error sized by
        the measured gate error rate, and
      - thermal relaxation FOR THE GATE'S OWN DURATION: a fixed-duration
        T1/T2 channel (20 ns for single-qubit gates, 60 ns for CZ),
    composed into a single error per gate. Also covers readout error.

    Why gate relaxation lives HERE (resident in the NoiseModel) rather than in
    the duration-aware pass: gate durations are fixed, so the relaxation per
    (gate, qubit) is fixed and can be a normal noise-model entry -- exactly how
    Aer's NoiseModel.from_backend does it. Keeping it resident means a qubit
    with T2 > T1 (whose thermal channel is a genuine non-unitary Kraus map)
    is present in the model at construction, so Aer precomputes its canonical
    Kraus and statevector simulation samples it per shot and scales to large
    qubit counts. The duration-aware pass (build_relaxation_pass) then only has
    to handle the part with VARIABLE duration: idle "delay" steps.

    Does NOT add relaxation to the virtual rz (zero time) or to measurement
    (readout error already models the measurement outcome).

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
    # Gate relaxation needs T1/T2 (seconds) and the gate durations (seconds).
    t1s_s, t2s_s = _per_qubit_t1_t2_seconds(cal, selected, t2_mode)
    sg_s, cz_s = sg_ns * 1e-9, cz_ns * 1e-9

    noise_model = NoiseModel()
    warnings: list[str] = []

    # --- Single-qubit gate: control error + gate-duration relaxation ---
    # When a single-qubit gate runs, two things happen: the applied rotation is
    # slightly off (control error -- a depolarizing error sized by the measured
    # single_gate_error), and the qubit decoheres for the gate's duration
    # (T1/T2 relaxation). We combine both into ONE error per qubit and attach it
    # to the timed native 1q gates. The virtual rz (zero time) gets only the
    # control error -- no relaxation, since it takes no time.
    for qname, qdata in selected:
        i = name_to_idx[qname]
        sg_err = qdata.get("single_gate_error", 0.001)
        if qdata.get(t2k, 99.0) < _T2_WARN_US:
            warnings.append(
                f"qubit {qname}: {t2_mode} T2={qdata.get(t2k):.3f}us "
                f"(<{_T2_WARN_US}us) -- heavy dephasing"
            )
        depol1 = depolarizing_error(sg_err, 1) if sg_err > 0 else None
        relax1 = _thermal_1q(t1s_s[i], t2s_s[i], sg_s)
        timed_err = _combine_control_relax(depol1, relax1)
        if timed_err is not None:
            noise_model.add_quantum_error(
                timed_err, _TIMED_1Q_GATES, [i], warnings=False
            )
        if depol1 is not None:
            noise_model.add_quantum_error(
                depol1, _VIRTUAL_1Q_GATES, [i], warnings=False
            )

    # --- Two-qubit (CZ) gate: control error + gate-duration relaxation ---
    # Same idea for CZ: a 2q depolarizing control error (sized by cz_error)
    # combined with thermal relaxation on BOTH qubits for the CZ duration. We
    # attach it in both qubit orderings ([i,j] and [j,i]) because the transpiler
    # may emit the CZ in either direction; the relaxation operands are tensored
    # in the matching order so they line up with each ordering's qubits.
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
        depol2 = depolarizing_error(cz_err, 2) if cz_err > 0 else None
        err_ij = _combine_control_relax(
            depol2, _thermal_nq([i, j], t1s_s, t2s_s, cz_s)
        )
        err_ji = _combine_control_relax(
            depol2, _thermal_nq([j, i], t1s_s, t2s_s, cz_s)
        )
        if err_ij is not None:
            noise_model.add_quantum_error(
                err_ij, _NATIVE_2Q_GATES, [i, j], warnings=False
            )
        if err_ji is not None:
            noise_model.add_quantum_error(
                err_ji, _NATIVE_2Q_GATES, [j, i], warnings=False
            )

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
    target=None,
):
    """Build the IDLE decoherence (T1/T2 on delays) as a duration-aware pass.

    Gate-time relaxation is now resident in the static noise model (see
    build_control_readout_noise_model), because gate durations are fixed. This
    pass handles the remaining, VARIABLE-duration part: the idle "delay" steps
    the scheduler inserts while a qubit waits for its neighbours. A long idle
    wait decoheres more than a short one -- the effect the IQM fake backends
    lack entirely (they never decohere idle qubits).

    How the pass is used: it is registered as a NoiseModel custom noise pass
    (noise_model._custom_noise_passes), exactly as Aer's NoiseModel.from_backend
    registers ITS delay-relaxation pass. Aer runs it at assemble time on any
    circuit that contains a Delay, and harmlessly skips circuits with none (a
    circuit with no idle gaps has no idle relaxation to add).

    Duration resolution: for a "delay" the pass reads the delay's own duration
    off the scheduled circuit. The durations are in dt ticks with one tick =
    1 ns (dt = 1e-9 s); passing dt_seconds = 1e-9 lets the pass convert N ticks
    into N * 1e-9 seconds before comparing to T1/T2 in seconds, so a 100 ns idle
    gap becomes 100 ns of decay. The zero-duration guard (below) drops any
    zero-length delay the padder may emit, which would otherwise build a
    degenerate channel.

    Args:
        target: the transpiler Target (carried through for consistency with the
            guard's duration resolution). Not strictly required for delays,
            which the pass reads off the circuit, but harmless to pass.

    Returns:
        (relaxation_pass, t1s_s, t2s_s) -- the pass plus the per-qubit T1/T2
        in seconds, indexed in the selected-qubit order, for logging.
    """
    from qiskit_aer.noise import RelaxationNoisePass
    from qiskit.circuit import Delay

    cal = _load_calibration(calibration_path)
    selected = _select_qubits(cal, num_qubits)

    # Calibration stores T1/T2 in microseconds; the pass wants seconds.
    t1s_s, t2s_s = _per_qubit_t1_t2_seconds(cal, selected, t2_mode)

    # Idle delays ONLY -- gate relaxation is resident in the static model.
    op_types = [Delay]
    kwargs = dict(t1s=t1s_s, t2s=t2s_s, op_types=op_types)
    if dt_seconds is not None:
        kwargs["dt"] = dt_seconds
    if target is not None:
        kwargs["target"] = target

    relax_pass = _build_zero_safe_relaxation_pass(RelaxationNoisePass, kwargs)
    return relax_pass, t1s_s, t2s_s


def _build_zero_safe_relaxation_pass(RelaxationNoisePass, kwargs):
    """Construct a RelaxationNoisePass that never builds a zero-duration
    channel.

    Root cause this guards against: Aer's RelaxationNoisePass resolves each
    operation's duration and calls thermal_relaxation_error(t1, t2, duration).
    The builds we target only skip ops whose resolved duration is *None* --
    they do NOT skip a resolved duration of *zero*. A zero-duration thermal
    relaxation channel is degenerate, and Aer raises "QuantumError: Kraus is
    empty" when it tries to apply it at simulation time. Zero-duration ops
    arise legitimately and from more than one source -- a virtual rz gate
    (genuinely zero time), or a zero-length Delay the scheduler can emit when
    a qubit has no idle gap -- so patching individual gate durations or
    stripping specific instructions is whack-a-mole. The correct, general fix
    is: whenever an op's duration resolves to zero, add no channel for it.

    Implementation: subclass the pass and override the per-op error builder to
    (1) resolve the duration the SAME way the parent does -- target lookup for
    gates, op.duration for delays / no-target -- and (2) return None (no
    channel) when that duration is zero. For every other case it defers
    entirely to the parent, so behaviour is otherwise identical.

    The duration-resolution logic below is replicated to match Aer's
    RelaxationNoisePass._thermal_relaxation_error. If a future Aer changes
    that method's name or shape, the subclass would no longer guard correctly
    -- so we verify at construction that the parent still has the expected
    internals, and if not, fall back to the stock pass and emit a clear
    warning rather than silently shipping unguarded (degenerate) channels.
    """
    import warnings as _warnings
    import numpy as _np
    from qiskit.circuit import Delay as _Delay

    # Safety check: the guard relies on these parent internals. If any is
    # missing (e.g. a future Aer refactor), do not pretend to guard.
    required = ("_thermal_relaxation_error", "_target", "_dt")
    probe_ok = True
    try:
        probe = RelaxationNoisePass(**kwargs)
        for attr in required:
            if not hasattr(probe, attr):
                probe_ok = False
                break
    except Exception:
        probe_ok = False

    if not probe_ok:
        _warnings.warn(
            "device_noise: could not verify RelaxationNoisePass internals "
            "needed for the zero-duration guard; falling back to the stock "
            "pass. Zero-duration ops (e.g. virtual rz, zero-length delays) "
            "may cause 'Kraus is empty' errors at simulation time. If you see "
            "that error, the guard needs updating for this Aer version.",
            RuntimeWarning,
        )
        return RelaxationNoisePass(**kwargs)

    def _resolve_duration_seconds(pass_obj, op, qubits):
        """Replicate the parent's duration resolution, returning seconds (or
        None if the parent would find no duration)."""
        from qiskit.utils.units import apply_prefix  # local import
        dt = pass_obj._dt
        target = pass_obj._target
        if target is not None:
            if op.name == "delay":
                duration = op.duration
                if duration is None:
                    return None
                if op.unit == "dt":
                    if dt is None:
                        return None
                    return op.duration * dt
                return apply_prefix(op.duration, op.unit)
            op_props = target.get(op.name)
            if op_props is not None:
                inst_props = op_props.get(tuple(qubits))
                if inst_props is not None:
                    return getattr(inst_props, "duration", None)
            return None
        # No target: read straight off the op.
        duration = getattr(op, "duration", None)
        if duration is None:
            return None
        if getattr(op, "unit", "dt") == "dt":
            if dt is None:
                return None
            return op.duration * dt
        return apply_prefix(op.duration, op.unit)

    class _ZeroSafeRelaxationNoisePass(RelaxationNoisePass):
        def _thermal_relaxation_error(self, op, qubits):
            dur = _resolve_duration_seconds(self, op, qubits)
            if dur is not None and dur == 0:
                # Zero-duration op: no decoherence, and building a channel
                # here would be degenerate. Skip it.
                return None
            # Otherwise defer entirely to the stock implementation, so all
            # other behaviour (None-duration warning, 1q/2q channel building,
            # inf-T1/T2 handling) is byte-identical to upstream.
            return super()._thermal_relaxation_error(op, qubits)

    try:
        return _ZeroSafeRelaxationNoisePass(**kwargs)
    except Exception:
        _warnings.warn(
            "device_noise: zero-duration-guarded RelaxationNoisePass could "
            "not be constructed; falling back to the stock pass.",
            RuntimeWarning,
        )
        return RelaxationNoisePass(**kwargs)

