# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Workstream A — placement_diversity: parse, scope-gate, disjoint selection,
device-max (same-ordering oracle), non-degeneracy, and the F5a-oracle
placement-equivalence integration.

Layers (mirrors the extraction test's offline/LUMI split):

  * Pure-Python (runs anywhere): the ``select_disjoint_placements`` walk, the
    ``count: auto`` device-max cross-checked against a SEPARATELY-WRITTEN
    fidelity-ranked greedy over the SAME ordering (RED-RULING-WORKSTREAM-A §3 —
    independence is in the implementation, not the algorithm; a different
    ordering computes a different device-max and would fail a correct
    implementation), the int-count fail-loud, and non-degeneracy.

  * Calibration-backed (skips cleanly without rustworkx/cal): the device-max on
    the real Q50 cal, computed and cross-checked against the same-ordering twin.

  * Parse/scope-gate (pure): placement_diversity parses; manual + diversity
    raises at parse; non-BYO + diversity raises.

The end-to-end placement-equivalence test (run the disjoint-selected set as one
multi-placement device-cal experiment and each chain isolated; assert byte
identity per subtree, reusing the F5a no-cross-talk CI guard as the oracle) runs
on LUMI inside the validation harness, not here — it needs aer/h5py and a full
sweep. This file locks down the selection logic and the carry/parse seams that
are checkable offline. See ``tests/f5a_no_crosstalk_validation.py`` for the
byte-identity oracle this composes with.
"""

from __future__ import annotations

import os

from lumi_hpc_qc.sweep.placement_solver import (
    Placement,
    PlacementDiversityConfig,
    placement_internal_edges,
    select_disjoint_placements,
)


# ─────────────────────────────────────────────────────────────────────────
# Helpers: a stub cal + Placement factory; an independent SAME-ORDERING oracle.
# ─────────────────────────────────────────────────────────────────────────

class _StubCal:
    def __init__(self, adjacency):
        self.adjacency = adjacency


def _mk(phys, score):
    return Placement(
        placement_id=min(phys),
        device_id="test_dev",
        device_prefix="test",
        qubit_mapping={i: f"QB{q}" for i, q in enumerate(phys)},
        physical_indices=sorted(phys),
        score=score,
        internal_edges=0,
        avg_readout_fidelity=score,
        avg_gate_fidelity=score,
    )


def _independent_disjoint_oracle(candidates, cal, *, count="auto", max_overlap=0):
    """A separately-written fidelity-ranked greedy over the SAME ordering.

    Does NOT import or call ``select_disjoint_placements``. Walks the candidate
    list in the order given (the caller pre-sorts by fidelity, identical to the
    selector) and accepts on the same disjoint rule. The independence is in this
    being a second hand-written implementation; the ORDERING is deliberately
    identical, because "device-max" is ordering-dependent and an oracle that
    walked a different order (connectivity-first, DSatur) would compute a
    DIFFERENT device-max and fail a correct selector (RED-RULING §3).
    """
    chosen = []
    qubits = set()
    edges = set()
    for p in candidates:
        pq = set(p.physical_indices)
        if pq.intersection(qubits):
            continue
        pe = placement_internal_edges(p.physical_indices, cal)
        if len(pe.intersection(edges)) > max_overlap:
            continue
        chosen.append(p)
        qubits.update(pq)
        edges.update(pe)
        if count != "auto" and len(chosen) == count:
            break
    return chosen


# ─────────────────────────────────────────────────────────────────────────
# Selection algorithm (pure).
# ─────────────────────────────────────────────────────────────────────────

def test_disjoint_picks_independent_not_clustered():
    cal = _StubCal({
        0: {1}, 1: {0, 2}, 2: {1},          # chain A: 0-1-2
        10: {11}, 11: {10, 12}, 12: {11},   # chain B: 10-11-12 (disjoint from A)
    })
    # Top score is A; second-best CLUSTERS with A (shares qubit 1); third is B.
    cands = [
        _mk([0, 1, 2], score=0.99),
        _mk([1, 2], score=0.98),            # overlaps A on {1,2} -> rejected
        _mk([10, 11, 12], score=0.90),
    ]
    sel = select_disjoint_placements(cands, cal, count="auto", max_overlap=0)
    sets = [tuple(p.physical_indices) for p in sel]
    assert sets == [(0, 1, 2), (10, 11, 12)]  # clustered middle one skipped


def test_count_auto_is_device_max_matches_same_ordering_oracle():
    cal = _StubCal({
        0: {1}, 1: {0}, 2: {3}, 3: {2}, 4: {5}, 5: {4},
    })
    cands = sorted(
        [_mk([0, 1], 0.9), _mk([2, 3], 0.8), _mk([4, 5], 0.7),
         _mk([1, 2], 0.85)],   # 1-2 not adjacent here -> but shares qubits 1,2
        key=lambda p: (-p.score, p.physical_indices),
    )
    sel = select_disjoint_placements(cands, cal, count="auto", max_overlap=0)
    oracle = _independent_disjoint_oracle(cands, cal, count="auto", max_overlap=0)
    assert [p.physical_indices for p in sel] == [p.physical_indices for p in oracle]
    # device-max is computed, never a literal:
    assert len(sel) == len(oracle)


def test_count_int_fail_loud_when_device_cannot_supply():
    cal = _StubCal({0: {1}, 1: {0}, 2: {3}, 3: {2}})  # at most 2 disjoint pairs
    cands = [_mk([0, 1], 0.9), _mk([2, 3], 0.8)]
    # Asking for 3 disjoint when only 2 fit must raise, not return 2 silently.
    raised = False
    try:
        select_disjoint_placements(cands, cal, count=3, max_overlap=0)
    except ValueError as e:
        raised = True
        assert "only 2" in str(e) and "family-collapse" in str(e)
    assert raised, "expected fail-loud on under-supply, got silent short count"


def test_count_int_exact_when_available():
    cal = _StubCal({0: {1}, 1: {0}, 2: {3}, 3: {2}, 4: {5}, 5: {4}})
    cands = [_mk([0, 1], 0.9), _mk([2, 3], 0.8), _mk([4, 5], 0.7)]
    sel = select_disjoint_placements(cands, cal, count=2, max_overlap=0)
    assert len(sel) == 2
    assert [p.physical_indices for p in sel] == [[0, 1], [2, 3]]  # top-2 disjoint


def test_max_overlap_budget_admits_shared_edge():
    # 0-1-2-3 path; placement A={0,1,2} (edges 0-1,1-2), B={2,3} shares qubit 2.
    # With a qubit overlap, B is rejected regardless of max_overlap (qubit
    # disjointness is absolute). Construct an edge-only overlap is impossible
    # without sharing a qubit on this topology, so assert the qubit rule holds
    # and that max_overlap does not loosen the QUBIT constraint.
    cal = _StubCal({0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}})
    cands = [_mk([0, 1, 2], 0.9), _mk([2, 3], 0.8)]
    sel = select_disjoint_placements(cands, cal, count="auto", max_overlap=5)
    assert [p.physical_indices for p in sel] == [[0, 1, 2]]  # B shares qubit 2


def test_non_degeneracy_scores_non_increasing_and_pairwise_disjoint():
    cal = _StubCal({0: {1}, 1: {0}, 2: {3}, 3: {2}, 4: {5}, 5: {4}})
    cands = sorted(
        [_mk([0, 1], 0.9), _mk([2, 3], 0.8), _mk([4, 5], 0.7)],
        key=lambda p: (-p.score, p.physical_indices),
    )
    sel = select_disjoint_placements(cands, cal, count="auto", max_overlap=0)
    # scores non-increasing in selection order
    for i in range(len(sel) - 1):
        assert sel[i].score >= sel[i + 1].score
    # pairwise disjoint (qubits)
    seen = set()
    for p in sel:
        s = set(p.physical_indices)
        assert not (s & seen)
        seen |= s


# ─────────────────────────────────────────────────────────────────────────
# Config / dataclass semantics (pure).
# ─────────────────────────────────────────────────────────────────────────

def test_diversity_config_no_crosstalk_flag():
    assert PlacementDiversityConfig(strategy="disjoint", max_overlap=0).no_crosstalk
    assert not PlacementDiversityConfig(strategy="disjoint", max_overlap=1).no_crosstalk


def test_diversity_config_default_inactive():
    pd = PlacementDiversityConfig()
    assert pd.strategy == "none"
    assert not pd.is_active()


# ─────────────────────────────────────────────────────────────────────────
# Parse + scope-gate (pure; exercises parse_sweep_config).
# ─────────────────────────────────────────────────────────────────────────

def _parse(yaml_dict):
    from lumi_hpc_qc.sweep.sweep_engine import parse_sweep_config
    return parse_sweep_config(yaml_dict)


def _byo_exp(**over):
    base = {
        "type": "byo_circuit",
        "circuit_script": "examples/byo/floquet_dtc_echo.py",
        "qubit_sizes": [10],
        "seed_list": [0, 1],
    }
    base.update(over)
    return base


def test_parse_diversity_block():
    cfg = _parse({"sweep": {"experiments": [
        _byo_exp(placement_diversity={"strategy": "disjoint",
                                      "max_overlap": 0, "count": "auto"}),
    ]}})
    pd = cfg.experiments[0].placement_diversity
    assert pd.strategy == "disjoint" and pd.max_overlap == 0 and pd.count == "auto"
    assert pd.is_active()


def test_parse_diversity_default_inactive_when_absent():
    cfg = _parse({"sweep": {"experiments": [_byo_exp()]}})
    assert not cfg.experiments[0].placement_diversity.is_active()


def test_parse_manual_plus_diversity_raises():
    raised = False
    try:
        _parse({"sweep": {"experiments": [
            _byo_exp(physical_qubits=[["QB8", "QB16"]],
                     placement_diversity={"strategy": "disjoint"}),
        ]}})
    except ValueError as e:
        raised = True
        assert "mutually exclusive" in str(e)
    assert raised, "manual + diversity must raise at parse (§9.1)"


def test_parse_bad_strategy_raises():
    raised = False
    try:
        _parse({"sweep": {"experiments": [
            _byo_exp(placement_diversity={"strategy": "spread_out"}),
        ]}})
    except ValueError as e:
        raised = True
        assert "strategy" in str(e)
    assert raised


def test_parse_bad_count_raises():
    for bad in (0, -1, "many", 2.5):
        raised = False
        try:
            _parse({"sweep": {"experiments": [
                _byo_exp(placement_diversity={"strategy": "disjoint",
                                              "count": bad}),
            ]}})
        except ValueError:
            raised = True
        assert raised, f"count={bad!r} must raise"


# ─────────────────────────────────────────────────────────────────────────
# Calibration-backed device-max (skips without rustworkx/cal).
# ─────────────────────────────────────────────────────────────────────────

def _load_q50_solver():
    try:
        import rustworkx  # noqa: F401
        from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
        from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    except Exception:
        return None, None
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    for name in ("q50_calibration_20260524_08c3c70f.json",
                 "q50_calibration_20260330.json"):
        cand = os.path.join(root, "examples", name)
        if os.path.exists(cand):
            cal = IQMv2Adapter().load(cand)
            solver = GeneralPlacementSolver()
            solver.add_device(cal)
            return solver, cal
    return None, None


def test_device_max_on_real_cal_matches_same_ordering_oracle():
    solver, cal = _load_q50_solver()
    if solver is None:
        import warnings
        warnings.warn("rustworkx or Q50 cal unavailable; skipping device-max layer")
        return

    # A 4q linear chain; full ranked candidate list (the selector's input).
    chain_edges = [(0, 1), (1, 2), (2, 3)]
    cands = solver.find_all_placements(
        circuit_edges=chain_edges, circuit_qubits=4, strategy="max_fidelity",
    )
    assert len(cands) > 10

    sel = select_disjoint_placements(cands, cal, count="auto", max_overlap=0)
    oracle = _independent_disjoint_oracle(cands, cal, count="auto", max_overlap=0)

    # Device-max is computed and equals the same-ordering twin — no literal.
    assert len(sel) == len(oracle)
    assert [p.physical_indices for p in sel] == [p.physical_indices for p in oracle]
    # And it is genuinely disjoint + fidelity-ordered.
    seen = set()
    for i, p in enumerate(sel):
        s = set(p.physical_indices)
        assert not (s & seen)
        seen |= s
        if i:
            assert sel[i - 1].score >= p.score
    print(f"       device-max (fidelity-ranked disjoint, 4q chain): {len(sel)}")


if __name__ == "__main__":
    import sys
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
        except AssertionError as e:
            print(f"  [FAIL] {fn.__name__}: {e}")
            failed += 1
        except Exception as e:  # noqa: BLE001
            print(f"  [ERROR] {fn.__name__}: {type(e).__name__}: {e}")
            failed += 1
    if failed:
        print(f"\nPLACEMENT-DIVERSITY: {failed} FAILURE(S)")
        sys.exit(1)
    print("\nPLACEMENT-DIVERSITY: ALL CHECKS PASSED")
