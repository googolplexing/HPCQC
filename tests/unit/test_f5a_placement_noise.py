# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""F5a — per-placement device-calibrated noise composition.

Verifies the D3.2 seam: a placement's physical qubits compose the
device-calibrated noise model (statevector path), instead of the historical
fidelity-driven self-selection. The seam lives in:
  - noise_model._resolve_selected      (the placement<->autoselect switch)
  - device_noise.build_control_readout_noise_model(physical_qubits=...)
  - device_noise.build_relaxation_pass(physical_qubits=...)
  - prepare.prepare_simulation(physical_qubits=..., physical_edges=...)

Pure-Python tests run anywhere; the builder tests need qiskit-aer (in-container
on LUMI) and importorskip otherwise.

Reference template: twin_simulator.build_placement_noise_model (the
density_matrix analogue, already placement-aware).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lumi_hpc_qc.backends.noise_model import (
    _resolve_selected,
    _select_qubits,
    _load_calibration,
)

# Repo-root-relative: tests/unit/<this> -> parents[2] == repo root.
CALIBRATION = str(
    Path(__file__).resolve().parents[2]
    / "examples" / "q50_calibration_20260524_08c3c70f.json"
)

# Real qubits / CZ edges from the committed Q50 calibration (verified present).
PLACEMENT_A = ["QB6", "QB5", "QB2", "QB1"]
PLACEMENT_B = ["QB3", "QB4", "QB7", "QB8"]
REAL_EDGE = ["QB1", "QB2"]          # a calibrated two-qubit gate
NONEDGE = ["QB6", "QB1"]            # both in A, but not a calibrated CZ pair
# QB35: the one qubit in the t1<t2<=2t1 regime (D2). Used to show its real
# (unclamped here) T2 threads through per-placement.
PLACEMENT_Q35 = ["QB35", "QB1", "QB2", "QB5"]


def _cal():
    return _load_calibration(CALIBRATION)


# ----------------------- pure: _resolve_selected -------------------------

def test_resolve_none_matches_autoselect():
    """physical_qubits=None reproduces _select_qubits byte-for-byte."""
    cal = _cal()
    got = _resolve_selected(cal, 4, None)
    expected = _select_qubits(cal, 4)
    assert [n for n, _ in got] == [n for n, _ in expected]


def test_resolve_uses_given_qubits_in_order():
    """A placement selects exactly those qubits, in logical order."""
    cal = _cal()
    got = _resolve_selected(cal, 4, PLACEMENT_A)
    assert [n for n, _ in got] == PLACEMENT_A
    # qdata is the calibration's own per-qubit dict (identity, not a copy mangle)
    assert got[0][1] is cal["qubits"]["QB6"]


def test_resolve_fail_loud_length():
    cal = _cal()
    with pytest.raises(ValueError, match="must match"):
        _resolve_selected(cal, 4, ["QB6", "QB5", "QB2"])      # 3 != 4


def test_resolve_fail_loud_unknown_name():
    cal = _cal()
    with pytest.raises(ValueError, match="not in calibration"):
        _resolve_selected(cal, 4, ["QB6", "QB5", "QB2", "QB9999"])


def test_resolve_edge_membership_ok():
    """A real calibrated edge among the placement passes."""
    cal = _cal()
    got = _resolve_selected(cal, 4, PLACEMENT_A, physical_edges=[REAL_EDGE])
    assert [n for n, _ in got] == PLACEMENT_A


def test_resolve_fail_loud_noncalibrated_edge():
    cal = _cal()
    with pytest.raises(ValueError, match="not a calibrated"):
        _resolve_selected(cal, 4, PLACEMENT_A, physical_edges=[NONEDGE])


def test_resolve_fail_loud_edge_outside_placement():
    cal = _cal()
    with pytest.raises(ValueError, match="outside"):
        _resolve_selected(cal, 4, PLACEMENT_A, physical_edges=[["QB1", "QB7"]])


# --------------- builders: need qiskit-aer (in-container) ----------------

def test_control_readout_keyed_to_placement():
    """The control/readout noise model is composed from the placement's
    qubits, and a different placement yields a different composition."""
    pytest.importorskip("qiskit_aer")
    from lumi_hpc_qc.backends.device_noise import (
        build_control_readout_noise_model,
    )
    _, _, info_a = build_control_readout_noise_model(
        CALIBRATION, num_qubits=4, physical_qubits=PLACEMENT_A,
    )
    _, _, info_b = build_control_readout_noise_model(
        CALIBRATION, num_qubits=4, physical_qubits=PLACEMENT_B,
    )
    assert info_a["selected_qubits"] == PLACEMENT_A
    assert info_b["selected_qubits"] == PLACEMENT_B
    assert info_a["selected_qubits"] != info_b["selected_qubits"]


def test_control_readout_none_reproduces_autoselect():
    """No placement -> historical self-selection, unchanged."""
    pytest.importorskip("qiskit_aer")
    from lumi_hpc_qc.backends.device_noise import (
        build_control_readout_noise_model,
    )
    _, _, info = build_control_readout_noise_model(CALIBRATION, num_qubits=4)
    expected = [n for n, _ in _select_qubits(_cal(), 4)]
    assert info["selected_qubits"] == expected


def test_relaxation_pass_threads_placement_t1_t2():
    """build_relaxation_pass returns per-qubit T1/T2 for the placement's
    qubits, in order — including QB35 (the t1<t2<=2t1 qubit)."""
    pytest.importorskip("qiskit_aer")
    from lumi_hpc_qc.backends.device_noise import (
        build_relaxation_pass, _per_qubit_t1_t2_seconds,
    )
    _, t1s, t2s = build_relaxation_pass(
        CALIBRATION, num_qubits=4, physical_qubits=PLACEMENT_Q35,
    )
    cal = _cal()
    selected = [(q, cal["qubits"][q]) for q in PLACEMENT_Q35]
    exp_t1, exp_t2 = _per_qubit_t1_t2_seconds(cal, selected, "ramsey")
    assert t1s == exp_t1
    assert t2s == exp_t2
    # QB35 is index 0: its real (clamp-irrelevant here) values threaded through.
    assert abs(t1s[0] - 2.044280090531019e-6) < 1e-12


def test_both_builders_consistent_on_same_placement():
    """The two builders self-select independently; the same placement must
    give them the same per-qubit T1/T2 (the desync trap the seam closes)."""
    pytest.importorskip("qiskit_aer")
    from lumi_hpc_qc.backends.device_noise import (
        build_relaxation_pass, _per_qubit_t1_t2_seconds,
    )
    cal = _cal()
    selected = [(q, cal["qubits"][q]) for q in PLACEMENT_A]
    ctrl_t1, ctrl_t2 = _per_qubit_t1_t2_seconds(cal, selected, "ramsey")
    _, relax_t1, relax_t2 = build_relaxation_pass(
        CALIBRATION, num_qubits=4, physical_qubits=PLACEMENT_A,
    )
    assert relax_t1 == ctrl_t1
    assert relax_t2 == ctrl_t2


def test_prepare_simulation_end_to_end_placement():
    """Full seam: prepare_simulation forwards the placement and routes the
    circuit onto it (identity initial_layout) without raising."""
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from lumi_hpc_qc.backends.prepare import prepare_simulation

    qc = QuantumCircuit(4)
    qc.x(0)
    qc.cz(0, 1)
    qc.measure_all()

    prep = prepare_simulation(
        [qc], "device-calibrated",
        calibration_path=CALIBRATION, num_qubits=4,
        physical_qubits=PLACEMENT_A, physical_edges=[REAL_EDGE],
        verbose=False,
    )
    assert prep.source == "device-calibrated"
    assert prep.info["selected_qubits"] == PLACEMENT_A
    assert len(prep.run_circuits) == 1


def test_prepare_simulation_none_path_unchanged():
    """No placement -> self-selection + free layout (the F4 baseline)."""
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from lumi_hpc_qc.backends.prepare import prepare_simulation

    qc = QuantumCircuit(4)
    qc.x(0)
    qc.cz(0, 1)
    qc.measure_all()

    prep = prepare_simulation(
        [qc], "device-calibrated",
        calibration_path=CALIBRATION, num_qubits=4, verbose=False,
    )
    expected = [n for n, _ in _select_qubits(_cal(), 4)]
    assert prep.info["selected_qubits"] == expected


# ====================================================================
# §2.1 BEHAVIORAL index-alignment — the D3.2 gate (RED-RESP-D3.4C §2).
# ====================================================================
# test_relaxation_pass_threads_placement_t1_t2 (above) asserts only that the
# T1/T2 LIST is built in placement order: exp_t1 is _per_qubit_t1_t2_seconds
# over the SAME `selected` list, so `t1s == exp_t1` is a tautology about list
# construction (already covered by test_resolve_uses_given_qubits_in_order).
# It never proves the load-bearing §2.1 claim: that when the prepared
# statevector simulator runs the SCHEDULED circuit, idle decoherence on circuit
# index k actually LANDS on physical_qubits[k]'s T1/T2 — the one place D3's own
# design said relabeling could leak (RelaxationNoisePass duration resolution).
#
# These two tests close that gap behaviorally. They build an idle `Delay` on a
# chosen logical qubit, run it through prepare_simulation's statevector sim, and
# read where the decoherence landed from the measured counts:
#   1. swap test (full model): swapping the placement order moves the surviving
#      population with it — a placement-INDEPENDENT (mis-keyed / self-selected /
#      sorted) pass would give the SAME marginal at index k for both orders, so
#      the wide-margin ordinal bands here cannot pass under a mis-key.
#   2. magnitude test (thermal-only): the survival at index k matches that
#      qubit's exp(-t/T1) — "matches THAT qubit's T1", not another index's.
#
# Asymmetric placement: QB35 (T1 = 2.04 us) is markedly lossy, QB8 (T1 = 50.3
# us) near-ideal — a ~25x T1 ratio. A qubit prepared in |1> and idled for ~5 us
# survives at exp(-t/T1): ~0.09 for QB35 vs ~0.91 for QB8. The end qubits are
# the distinctive pair Red pointed at (PLACEMENT_Q35 already carries QB35), put
# at indices 0 and N-1 so the swap moves the signal across the whole register.

PLACEMENT_LOSSY_FIRST = ["QB35", "QB1", "QB2", "QB8"]   # logical 0 lossy, 3 ideal
PLACEMENT_IDEAL_FIRST = ["QB8", "QB1", "QB2", "QB35"]   # ends swapped

# dt = 1 ns in the device-calibrated path (prepare.py: dt_s = 1e-9), so a delay
# of 5000 dt ticks is 5 us of idle decoherence — long enough that the idle/delay
# RelaxationNoisePass (the §2.1 hazard site) dominates over the ~20 ns resident
# gate-time relaxation on the single X (the idle is >99% of the exposure).
_IDLE_DELAY_DT = 5000
_SIM_SEED = 1234          # pin the per-shot Kraus sampling so the test is stable
_SHOTS = 4096             # 3-sigma shot noise on a ~0.09 marginal is ~0.013


def _excited_idle_circuit():
    """4-qubit circuit: excite logical qubits 0 and 3 (X), idle them for the
    same long Delay, then measure all.

    Logical 1 and 2 stay in |0>, where amplitude damping is a no-op, so they
    carry no survival signal and cannot contaminate the ends. Only the excited
    ends (0 and 3) report a T1-survival fraction, and those are the indices the
    swap exchanges.
    """
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    qc.x(0)
    qc.x(3)
    qc.delay(_IDLE_DELAY_DT, 0, unit="dt")
    qc.delay(_IDLE_DELAY_DT, 3, unit="dt")
    qc.measure_all()
    return qc


def _p1(counts, qubit):
    """Marginal P(qubit == 1) from Aer counts.

    Aer is little-endian: in a bitstring, qubit 0 is the RIGHTMOST character, so
    bit for logical qubit k is bitstr[::-1][k]. (measure_all uses one creg, so
    keys have no spaces; strip defensively.)
    """
    total = sum(counts.values())
    ones = sum(
        c for b, c in counts.items() if b.replace(" ", "")[::-1][qubit] == "1"
    )
    return ones / total


def _run_idle(physical_qubits, *, spec=None):
    """Run the excited-idle circuit on the placement-keyed statevector sim,
    returning (counts, prep).

    Why the explicit Delay is NOT handed into prepare_simulation's transpile:
    the device-calibrated target basis has no `delay`. In production, idle
    Delays are inserted by the ALAP scheduler AFTER native translation, so an
    explicit Delay placed in the INPUT circuit dies in BasisTranslator
    (TranspilerError: cannot translate "delay"). Instead we build the prepared
    simulator + placement-keyed noise model from a gate-only seed (which
    transpiles fine; the noise model is built from the calibration + placement,
    not from the seed's contents), then run the explicit-Delay circuit DIRECTLY
    on prep.simulator. That is exactly how the idle relaxation pass is meant to
    be driven: it is a NoiseModel custom pass (op_types=[Delay], dt=1e-9) that
    Aer runs at simulate time on any circuit containing a Delay, reading each
    Delay's own duration. Running on the same simulator object keeps the
    placement-keyed t1s[k]/t2s[k] (= physical_qubits[k]) bound to circuit
    index k, which is the §2.1 claim under test.
    """
    from qiskit import QuantumCircuit
    from lumi_hpc_qc.backends.prepare import prepare_simulation

    seed = QuantumCircuit(4)
    seed.x(0)
    seed.measure_all()
    prep = prepare_simulation(
        [seed], "device-calibrated",
        calibration_path=CALIBRATION, num_qubits=4,
        physical_qubits=physical_qubits, spec=spec, verbose=False,
    )
    # The placement must have threaded into the noise model in logical order,
    # and the idle/delay relaxation pass must be attached, or this test is not
    # exercising what it claims to.
    assert prep.info["selected_qubits"] == physical_qubits
    assert prep.relaxation_active is True
    job = prep.simulator.run(
        _excited_idle_circuit(), shots=_SHOTS, seed_simulator=_SIM_SEED,
    )
    return job.result().get_counts(0), prep


def test_idle_relaxation_lands_on_placement_qubit_swap():
    """SWAP test (full model): the surviving excited population at logical index
    k tracks physical_qubits[k] across a placement swap.

    Under LOSSY_FIRST = [QB35, QB1, QB2, QB8]: index 0 = QB35 decays hard,
    index 3 = QB8 survives. Swapping to IDEAL_FIRST flips both ends. A
    placement-independent (mis-keyed) pass would leave index 0's marginal
    unchanged across the swap; the disjoint bands below (lossy < 0.40 < 0.60 <
    ideal) make that impossible to pass.
    """
    pytest.importorskip("qiskit_aer")
    counts_lf, _ = _run_idle(PLACEMENT_LOSSY_FIRST)   # idx0=QB35, idx3=QB8
    counts_if, _ = _run_idle(PLACEMENT_IDEAL_FIRST)   # idx0=QB8,  idx3=QB35

    p0_lf, p3_lf = _p1(counts_lf, 0), _p1(counts_lf, 3)
    p0_if, p3_if = _p1(counts_if, 0), _p1(counts_if, 3)

    # Within a placement: the lossy end is far below the ideal end.
    assert p0_lf < 0.40 < 0.60 < p3_lf, (p0_lf, p3_lf)   # QB35 vs QB8
    assert p3_if < 0.40 < 0.60 < p0_if, (p0_if, p3_if)   # QB35 vs QB8 (swapped)

    # Across the swap: index 0's survival moves from lossy to ideal (and index 3
    # the reverse). The mis-key signature would be p0_lf ~= p0_if; assert the
    # gap is decisive.
    assert p0_if - p0_lf > 0.40, (p0_lf, p0_if)
    assert p3_lf - p3_if > 0.40, (p3_lf, p3_if)
    # The same physical qubit (QB35) at the two different ends agrees, and so
    # does QB8 — the survival follows the qubit, not the index.
    assert abs(p0_lf - p3_if) < 0.15, (p0_lf, p3_if)      # both QB35
    assert abs(p3_lf - p0_if) < 0.15, (p3_lf, p0_if)      # both QB8


def test_idle_relaxation_magnitude_matches_qubit_t1():
    """MAGNITUDE test (thermal-only): the survival at each index matches that
    qubit's exp(-t/T1) — proving the decoherence is keyed to physical_qubits[k]'s
    own T1, not merely "some asymmetric value".

    Thermal-only (no readout flip, no depolarizing) so the measured marginal is
    the clean amplitude-damping survival. Idle exposure = 5000 ns delay (+ ~20 ns
    X gate, <0.4% of T1 for QB35), so exp(-t/T1) with the calibration's own T1.
    """
    pytest.importorskip("qiskit_aer")
    import math
    from lumi_hpc_qc.backends.noise_spec import NoiseSpec

    thermal_only = NoiseSpec(
        single_qubit_depolarizing=False, two_qubit_depolarizing=False,
        readout=False, thermal_relaxation=True,
    )
    counts, _ = _run_idle(PLACEMENT_LOSSY_FIRST, spec=thermal_only)

    cal = _cal()
    t_s = _IDLE_DELAY_DT * 1e-9          # delay only; gate adds <0.001 abs here
    exp_qb35 = math.exp(-t_s / (cal["qubits"]["QB35"]["t1_us"] * 1e-6))   # ~0.086
    exp_qb8 = math.exp(-t_s / (cal["qubits"]["QB8"]["t1_us"] * 1e-6))     # ~0.905

    p0, p3 = _p1(counts, 0), _p1(counts, 3)   # idx0 = QB35, idx3 = QB8
    # Tolerance absorbs 3-sigma shot noise (~0.013) + the ~20 ns gate term; 0.05
    # cannot confuse 0.086 with 0.905 (they are 0.82 apart).
    assert abs(p0 - exp_qb35) < 0.05, (p0, exp_qb35)
    assert abs(p3 - exp_qb8) < 0.05, (p3, exp_qb8)


# ====================================================================
# §2.1 integration: idle keying through the FULL transpile + ALAP path.
# ====================================================================
# Companion to test_idle_relaxation_lands_on_placement_qubit_swap, which drives
# an EXPLICIT Delay directly on the prepared simulator (clean, but it bypasses
# prepare_simulation's transpile + ALAP scheduler). This test creates the idle
# the way PRODUCTION does -- the qubit waits and the ALAP scheduler INSERTS the
# Delay -- and confirms that scheduler-inserted delay is decohered with the
# placement-keyed T1/T2 too.
#
# The hazard ALAP poses (see BLUE notes): ALAP front-loads idle onto the GROUND
# state, where relaxation is a no-op, so a naively-excited-then-idle qubit gets
# its excitation pushed late and never decoheres. We defeat that by trapping the
# target (logical 0) in |1> between TWO CZ anchors with the worker (logical 1):
# the worker runs a chain of `sx` gates between the anchors, so the double
# anchor removes ALL of ALAP's slack and the scheduler MUST pad the target's
# wait with a Delay equal to the worker-chain duration -- placed ON the excited
# state.
#
# Worker time-filler is `sx` with a `barrier(1)` after each one. A first cut
# used `id` because it is noiseless (in no _TIMED_1Q / _VIRTUAL_1Q / _NATIVE_2Q
# list) and leaves the worker in |0>. It failed: both placements measured idx0
# ~0.87, consistent with NO scheduler-inserted idle, only the ~140 ns of X +
# CZ + CZ gate-time relaxation. IGate.to_matrix() IS I, so RemoveIdentityEquivalent
# strips it even at opt 0. Switching to `sx` (matrix != I) failed the SAME way
# with the SAME 0.87 -- a 1q-fusion pass resynthesized 250 sx into one equivalent
# gate (sx^250 = X, 20 ns), collapsing the worker timeline to a single gate's
# worth of time. (Empirically, opt_level=0 does NOT disable 1q fusion in this
# qiskit; expect Optimize1qGatesDecomposition or similar to run regardless.)
# The reliable fix is mechanical: a `barrier(1)` after every `sx(1)`. Every 1q
# optimizer treats barriers as run boundaries, so each sx becomes a length-1
# run that resynthesizes to itself (sx is already minimal in this basis), and
# 250 individual sx gates survive into scheduling with 20 ns each. Barriers are
# zero-duration scheduling markers, so the worker still occupies 250 * 20 ns
# = 5 us. Trade vs `id`: sx is noised (depolarizing + gate-time thermal), so
# the worker decoheres during its chain. This DOES NOT affect target population:
# CZ is diagonal in the computational basis, so however the worker's diagonal
# looks at the CZ anchors (|0>, |1>, or any incoherent mixture), CZ_1 and CZ_2
# leave the target's |1> population invariant. The only thing that can move
# target P(1) is the thermal channel on the target's own scheduler-inserted
# Delay -- which is exactly the signal under test.
#
# Placement is a calibrated CZ edge with a ~17x T1 ratio -- QB35 (T1 = 2.04 us,
# lossy) and QB36 (T1 = 34.9 us, ideal). Worker chain = 250 * 20 ns = 5 us idle.

_WORKER_IDLE_GATES = 250          # * single_gate_time (20 ns) = ~5 us scheduled idle
PLACEMENT_LOSSY_TARGET = ["QB35", "QB36"]   # idx0 = QB35 (lossy); worker idx1 = QB36
PLACEMENT_IDEAL_TARGET = ["QB36", "QB35"]   # idx0 = QB36 (ideal); worker swapped


def _sandwiched_idle_circuit(n_idle):
    """Target (logical 0) excited and trapped in |1> between two CZ anchors
    while the worker (logical 1) runs `n_idle` `sx` gates between them, each
    followed by a `barrier(1)` so 1q-fusion passes cannot resynthesize the
    chain into a single equivalent gate. See the header above for why."""
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(2)
    qc.x(0)               # excite target -> |1>
    qc.cz(0, 1)           # early anchor: tie target to the worker's timeline
    for _ in range(n_idle):
        qc.sx(1)          # worker busy (20 ns, non-identity matrix); target waits in |1>
        qc.barrier(1)     # force each sx to be its own length-1 run -- 1q fusion passes treat
                          # barriers as run boundaries, so the chain cannot be resynthesized
                          # into a single equivalent unitary (sx^250 = X, which collapsed
                          # 250 * 20 ns of worker time to 20 ns in a previous iteration).
    qc.cz(0, 1)           # late anchor: target cannot proceed until the worker is done
    qc.measure_all()
    return qc


def _run_scheduled_idle(physical_qubits):
    """Run the sandwiched-idle circuit through the FULL device-calibrated path
    (transpile + ALAP scheduling), so the idle Delay is SCHEDULER-INSERTED, then
    decohered by the relaxation pass. Returns the counts."""
    from lumi_hpc_qc.backends.prepare import prepare_simulation

    prep = prepare_simulation(
        [_sandwiched_idle_circuit(_WORKER_IDLE_GATES)], "device-calibrated",
        calibration_path=CALIBRATION, num_qubits=2,
        physical_qubits=physical_qubits, optimization_level=0, verbose=False,
    )
    assert prep.info["selected_qubits"] == physical_qubits
    assert prep.relaxation_active is True
    job = prep.simulator.run(prep.run_circuits, shots=_SHOTS, seed_simulator=_SIM_SEED)
    return job.result().get_counts(0)


def test_idle_relaxation_tracks_placement_through_full_schedule():
    """SWAP test through the full transpile+ALAP path (no explicit Delay):
    proves the relaxation pass's T1/T2 keying tracks physical_qubits[k] -- the
    D3.2 §2.1 index-alignment claim -- not just that *some* T1/T2 is applied.

    Construction: target (logical idx0) is excited and trapped in |1> between
    two CZ anchors while the worker (logical idx1) runs a busy sx+barrier
    chain. ALAP scheduling makes the target's wire idle for the worker's chain
    duration (~5 us); PadDelay materializes that as a Delay on the target's
    wire; the relaxation pass decoheres it per the physical qubit at idx0.

    [QB35, QB36] -> idx0 = QB35 (T1 = 2.04 us) decays hard during the ~5 us
    scheduler-inserted idle; [QB36, QB35] -> idx0 = QB36 (T1 = 34.9 us)
    survives. Both placements run the SAME circuit; the only thing that changes
    is which physical qubit maps to logical idx0. The swap differential
    asserted below is the index-alignment proof: if the relaxation pass were
    keyed to logical index instead of physical qubit, both placements would
    produce comparable p0 values (same band) since the worker chain is
    identical in both. A non-trivial differential forces the conclusion that
    decoherence on logical-index-k is keyed to physical_qubits[k]'s T1/T2.
    Magnitude bands alone would not separate "lands on the right qubit" from
    "lands on a qubit with comparable T1/T2"; the swap differential does.

    Precondition: "delay" must be in _NATIVE_BASIS so PadDelay actually inserts
    the idle Delay -- otherwise PadDelay silently skips and the test sees only
    gate-time |1> exposure (~140 ns), no idle decay. See
    FINDING-PADDELAY-IDLE-NOT-INSERTED-v1_0.md.
    """
    pytest.importorskip("qiskit_aer")
    p0_lossy = _p1(_run_scheduled_idle(PLACEMENT_LOSSY_TARGET), 0)   # QB35 target
    p0_ideal = _p1(_run_scheduled_idle(PLACEMENT_IDEAL_TARGET), 0)   # QB36 target

    # Disjoint bands. ~5 us idle gives QB35 ~0.13 and QB36 ~0.85 (full model, with
    # readout); the 0.40/0.60 rails leave ~0.25 of margin on each side -- wide
    # enough to absorb the exact scheduled idle (the worker chain, ~5 us, plus
    # CZ/measure alignment) and 3-sigma shot noise (~0.016 at 4096 shots). The
    # assertion is deliberately ORDINAL, not a precise magnitude (the explicit-
    # Delay magnitude test above pins the absolute exp(-t/T1) value).
    assert p0_lossy < 0.40, p0_lossy
    assert p0_ideal > 0.60, p0_ideal
    # The §2.1 index-alignment proof: across the swap, idx0 survival moves
    # decisively with the physical qubit. A mis-keyed pass would fail this
    # rail regardless of which T1/T2 it picked, because the chosen T1/T2
    # cannot simultaneously give p0_lossy < 0.40 AND p0_ideal > 0.60.
    assert p0_ideal - p0_lossy > 0.40, (p0_lossy, p0_ideal)


def test_prepare_simulation_inserts_delays_in_scheduled_circuit():
    """REGRESSION GUARD for FINDING-PADDELAY-IDLE-NOT-INSERTED-v1_0.md.

    If "delay" is ever dropped from _NATIVE_BASIS (or otherwise omitted from
    the device-calibrated Target's operation_names), PadDelay silently skips
    Delay insertion on every qubit (BasePadding.__delay_supported returns
    False), and the scheduled circuit emerging from prepare_simulation will
    contain zero Delay instructions. The RelaxationNoisePass(op_types=
    [Delay]) is then starved of anything to act on, and idle-time
    decoherence is silently dropped from every device-calibrated run.

    The runtime precondition assertion in _prepare_device_calibrated is the
    loud-fail layer (raises RuntimeError immediately at Target construction
    time). This test is the structural regression guard at the test-suite
    layer: it builds a multi-gate circuit guaranteed to produce idle gaps
    under ALAP scheduling (X on target, CZ entangler, several sx on worker,
    second CZ anchor, measure), runs it through the full device-calibrated
    prepare path, and asserts at least one Delay survives into the prepared
    circuit. If this test fails, the silent-skip defect has been
    reintroduced by another route (e.g. a passmgr config that strips delays
    post-PadDelay, or a Target construction path that bypasses
    _NATIVE_BASIS).
    """
    pytest.importorskip("qiskit_aer")
    from qiskit import QuantumCircuit
    from lumi_hpc_qc.backends.prepare import prepare_simulation

    qc = QuantumCircuit(2, 2)
    qc.x(0)                  # target excited to |1>
    qc.cz(0, 1)              # entangler -- ties target timeline to worker
    for _ in range(5):
        qc.sx(1)             # worker busy gates create idle gap on target wire
        qc.barrier(1)
    qc.cz(0, 1)              # second anchor
    qc.measure([0, 1], [0, 1])

    prep = prepare_simulation(
        [qc], "device-calibrated",
        calibration_path=CALIBRATION, num_qubits=2,
        physical_qubits=PLACEMENT_LOSSY_TARGET, optimization_level=0,
        verbose=False,
    )
    scheduled = prep.run_circuits[0]
    delay_count = sum(
        1 for inst in scheduled.data if inst.operation.name == "delay"
    )
    assert delay_count > 0, (
        f"Scheduled device-calibrated circuit contains {delay_count} Delay "
        f"instructions; expected at least one. PadDelay must be silently "
        f"skipping insertion. Verify \"delay\" is in _NATIVE_BASIS in "
        f"src/lumi_hpc_qc/backends/prepare.py. "
        f"See FINDING-PADDELAY-IDLE-NOT-INSERTED-v1_0.md."
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
