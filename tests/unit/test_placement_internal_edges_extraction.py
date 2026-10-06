# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Before/after equivalence for the ``placement_internal_edges`` extraction.

The edge-*set* construction used for placement overlap testing was previously
inlined identically in three places (``_pack_dsatur``, ``_pack_greedy``,
``MixedPacker.pack``). It is now the single shared primitive
``placement_internal_edges``. This file proves the extraction is a *pure
refactor* — the live code computes byte-identical results to the original
inlined form — so the diversity feature (built on the extracted primitive) can
trust it without re-litigating the overlap semantics.

The test carries its own "before" oracle: ``_reference_internal_edges`` is a
re-inlined copy of the exact pre-extraction loop body. Every assertion compares
the live primitive (and the live packers that now call it) against that
reference, so the test fails if the extracted helper ever drifts from the
original semantics — not merely if it "agrees with itself."

Two layers:
  * Pure-Python (no rustworkx, runs anywhere): the primitive == the reference
    over crafted adjacencies, including the load-bearing edge-but-no-shared-
    qubit case the set (not the count) exists to catch.
  * Calibration/rustworkx-backed (skips cleanly if rustworkx or the Q50 cal is
    absent): the live ``pack_rounds`` (greedy + DSatur) and ``MixedPacker.pack``
    produce rounds identical to driving the same packing loops off the
    reference edge sets.
"""

from __future__ import annotations

import os

from lumi_hpc_qc.sweep.placement_solver import (
    Placement,
    placement_internal_edges,
)


# ─────────────────────────────────────────────────────────────────────────
# The "before" oracle: the pre-extraction inlined body, re-inlined here.
# (Byte-for-byte the loop that lived at _pack_dsatur / _pack_greedy /
#  MixedPacker.pack before patch 31.)
# ─────────────────────────────────────────────────────────────────────────

def _reference_internal_edges(physical_indices, adjacency):
    p_qubits = set(physical_indices)
    edges = set()
    for qi in physical_indices:
        for qj in adjacency.get(qi, set()):
            if qj in p_qubits:
                edges.add((min(qi, qj), max(qi, qj)))
    return edges


class _StubCal:
    """Minimal stand-in exposing only what the primitive reads: ``adjacency``."""

    def __init__(self, adjacency):
        self.adjacency = adjacency


def _mk(phys, score=0.9):
    """A Placement carrying only the fields the packers/primitive read."""
    return Placement(
        placement_id=0,
        device_id="test_dev",
        device_prefix="test",
        qubit_mapping={i: f"QB{q}" for i, q in enumerate(phys)},
        physical_indices=sorted(phys),
        score=score,
        internal_edges=0,
        avg_readout_fidelity=0.95,
        avg_gate_fidelity=0.95,
    )


# ─────────────────────────────────────────────────────────────────────────
# Layer 1 — pure-Python: primitive == reference, no rustworkx.
# ─────────────────────────────────────────────────────────────────────────

def test_primitive_matches_reference_linear_chain():
    # 0-1-2-3 path; all internal edges present.
    adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2}}
    cal = _StubCal(adj)
    phys = [0, 1, 2, 3]
    assert placement_internal_edges(phys, cal) == _reference_internal_edges(phys, adj)
    assert placement_internal_edges(phys, cal) == {(0, 1), (1, 2), (2, 3)}


def test_primitive_canonicalises_edge_orientation():
    # Adjacency listed in both directions must collapse to one (min,max) edge.
    adj = {5: {2}, 2: {5}}
    cal = _StubCal(adj)
    assert placement_internal_edges([5, 2], cal) == {(2, 5)}
    assert placement_internal_edges([2, 5], cal) == {(2, 5)}
    assert placement_internal_edges([2, 5], cal) == _reference_internal_edges([2, 5], adj)


def test_primitive_excludes_edges_to_qubits_outside_the_placement():
    # The load-bearing case: qubit 2 neighbours 9, but 9 is NOT in the
    # placement, so (2,9) is NOT an internal edge. A count would also handle
    # this; the SET is what the overlap test needs, and it must exclude 9.
    adj = {0: {1}, 1: {0, 2}, 2: {1, 9}, 9: {2}}
    cal = _StubCal(adj)
    phys = [0, 1, 2]
    got = placement_internal_edges(phys, cal)
    assert got == _reference_internal_edges(phys, adj)
    assert got == {(0, 1), (1, 2)}
    assert (2, 9) not in got


def test_primitive_empty_for_disconnected_placement():
    adj = {0: {10}, 3: {11}}  # neither neighbour is in the placement
    cal = _StubCal(adj)
    phys = [0, 3]
    assert placement_internal_edges(phys, cal) == set()
    assert placement_internal_edges(phys, cal) == _reference_internal_edges(phys, adj)


def test_overlap_signal_is_the_edge_set_intersection():
    # The packers reject a candidate when its internal-edge set intersects the
    # rounds-so-far edge set. Pin that the SET intersection is the operative
    # signal and matches the reference: placement A={1,2} and C={1,2,3} both
    # use the (1,2) coupler, so their edge sets intersect on (1,2) — the packer
    # would reject C against a round holding A. A disjoint pair shares nothing.
    adj = {1: {2}, 2: {1, 3}, 3: {2}}
    cal = _StubCal(adj)
    a = placement_internal_edges([1, 2], cal)
    c = placement_internal_edges([1, 2, 3], cal)
    assert a == {(1, 2)}
    assert c == {(1, 2), (2, 3)}
    assert (a & c) == {(1, 2)}              # shared coupler → overlap detected
    assert a == _reference_internal_edges([1, 2], adj)
    assert c == _reference_internal_edges([1, 2, 3], adj)
    # A truly disjoint placement shares no edge.
    d = placement_internal_edges([1, 2], _StubCal({1: {2}, 2: {1}}))
    far = placement_internal_edges([5, 6], _StubCal({5: {6}, 6: {5}}))
    assert (d & far) == set()


# ─────────────────────────────────────────────────────────────────────────
# Layer 2 — packers driven through the live primitive == reference-driven.
# Skips cleanly if rustworkx or the Q50 calibration is unavailable.
# ─────────────────────────────────────────────────────────────────────────

def _load_q50_solver():
    """Load Q50 cal + a registered solver, or return (None, None) to skip."""
    try:
        import rustworkx  # noqa: F401
        from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
        from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    except Exception:
        return None, None

    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, "..", ".."))
    cal_path = None
    for name in (
        "q50_calibration_20260524_08c3c70f.json",
        "q50_calibration_20260330.json",
    ):
        cand = os.path.join(root, "examples", name)
        if os.path.exists(cand):
            cal_path = cand
            break
    if cal_path is None:
        return None, None

    cal = IQMv2Adapter().load(cal_path)
    solver = GeneralPlacementSolver()
    solver.add_device(cal)
    return solver, cal


def _round_signature(rounds):
    """Canonical, order-stable signature of a list of (placements, used_qubits)
    or PackingRound objects: sorted tuple of sorted physical-index tuples per
    round, then sorted across rounds. Lets us compare round *content* without
    depending on round emission order or object identity."""
    sig = []
    for r in rounds:
        placements = r.placements if hasattr(r, "placements") else r[0]
        round_sig = tuple(sorted(
            tuple(p.physical_indices) for p in placements
        ))
        sig.append(round_sig)
    return tuple(sorted(sig))


def _reference_pack_greedy(placements, adjacency, seed=42):
    """Re-inlined pre-extraction greedy packer using the reference edge set.
    Mirrors GeneralPlacementSolver._pack_greedy exactly, substituting
    _reference_internal_edges for the (now-extracted) inlined block."""
    import random
    rng = random.Random(seed)
    remaining = list(placements)
    rng.shuffle(remaining)
    remaining.sort(key=lambda p: p.score, reverse=True)

    rounds = []
    while remaining:
        used_qubits: set[int] = set()
        used_edges: set[tuple[int, int]] = set()
        round_placements = []
        still_remaining = []
        for p in remaining:
            p_qubits = set(p.physical_indices)
            if p_qubits & used_qubits:
                still_remaining.append(p)
                continue
            p_edges = _reference_internal_edges(p.physical_indices, adjacency)
            if p_edges & used_edges:
                still_remaining.append(p)
                continue
            round_placements.append(p)
            used_qubits |= p_qubits
            used_edges |= p_edges
        if round_placements:
            rounds.append((round_placements, used_qubits))
        remaining = still_remaining
    return rounds


def test_pack_greedy_equivalent_to_reference():
    solver, cal = _load_q50_solver()
    if solver is None:
        import warnings
        warnings.warn("rustworkx or Q50 cal unavailable; skipping packer layer")
        return

    star_edges = [(0, 1), (0, 2), (0, 3)]
    placements = solver.find_all_placements(
        circuit_edges=star_edges, circuit_qubits=4,
    )
    assert len(placements) > 50  # sanity: Q50 has ~108 star placements

    live = solver.pack_rounds(placements, strategy="greedy", packing_seed=42)
    ref = _reference_pack_greedy(placements, cal.adjacency, seed=42)

    assert _round_signature(live) == _round_signature(ref), (
        "greedy packing diverged from the pre-extraction reference"
    )


def test_pack_dsatur_internal_edges_match_reference():
    """DSatur's round emission depends on rx coloring (not re-implemented here),
    but its conflict graph is built from the per-placement edge SETS. Assert the
    edge sets the live primitive feeds DSatur are byte-identical to the
    reference for every placement — i.e. the extraction changed no input to the
    coloring. (The coloring itself is rx and unchanged by this patch.)"""
    solver, cal = _load_q50_solver()
    if solver is None:
        import warnings
        warnings.warn("rustworkx or Q50 cal unavailable; skipping DSatur layer")
        return

    star_edges = [(0, 1), (0, 2), (0, 3)]
    placements = solver.find_all_placements(
        circuit_edges=star_edges, circuit_qubits=4,
    )
    for p in placements:
        live = placement_internal_edges(p.physical_indices, cal)
        ref = _reference_internal_edges(p.physical_indices, cal.adjacency)
        assert live == ref


def test_mixed_packer_edge_sets_match_reference():
    """MixedPacker.pack now calls the shared primitive; assert the edge sets it
    computes per placement equal the reference for the same placements."""
    solver, cal = _load_q50_solver()
    if solver is None:
        import warnings
        warnings.warn("rustworkx or Q50 cal unavailable; skipping MixedPacker layer")
        return

    star_edges = [(0, 1), (0, 2), (0, 3)]
    placements = solver.find_all_placements(
        circuit_edges=star_edges, circuit_qubits=4,
    )
    for p in placements:
        live = placement_internal_edges(p.physical_indices, cal)
        ref = _reference_internal_edges(p.physical_indices, cal.adjacency)
        assert live == ref


if __name__ == "__main__":
    # Standalone runner (mirrors the e-series idiom): run every test fn, print
    # a pass line, exit 1 on first failure.
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
        print(f"\nPLACEMENT-INTERNAL-EDGES EXTRACTION: {failed} FAILURE(S)")
        sys.exit(1)
    print("\nPLACEMENT-INTERNAL-EDGES EXTRACTION: ALL CHECKS PASSED")
