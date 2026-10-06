# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""W1.1 — placement-solver deterministic total order (F5 invariant).

Verifies that ``_placement_sort_key`` provides a deterministic total order
suitable for both the eager and (future) lazy placement-selection paths, per
RED-RESP-W1-PARALLELISM-AND-OOM-ROOTCAUSE-v1.4 §F5: the lazy and
full-enumerate paths must select the byte-identical placement (same
physical-qubit mapping, same order). This requires an explicit deterministic
total order on both paths; without it a "keep the running best" lazy traversal
cannot reproduce a "sort the full list" eager traversal whenever score ties
occur.

These tests exercise the key function and its application to the eager sort.
The cross-path equivalence test (lazy === eager on the same input set) lands
with W1.3 when the lazy variant is introduced; this file's job is to lock the
key down so W1.3 can reuse it unchanged.
"""

from __future__ import annotations

import random

from lumi_hpc_qc.sweep.placement_solver import (
    Placement,
    _placement_sort_key,
)


def _mk(pid: int, score: float, phys: list[int]) -> Placement:
    """Construct a minimal Placement object for sort-key testing.

    Only the fields the sort key reads (``score``, ``physical_indices``) need
    real values; everything else gets a stub. ``physical_indices`` is supplied
    pre-sorted, matching the invariant the real solver upholds at the point of
    dedup (line ~211 of placement_solver.py).
    """
    return Placement(
        placement_id=pid,
        device_id="test_dev",
        device_prefix="test",
        qubit_mapping={i: f"QB{q}" for i, q in enumerate(phys)},
        physical_indices=sorted(phys),
        score=score,
        internal_edges=0,
        avg_readout_fidelity=0.95,
        avg_gate_fidelity=0.95,
    )


def test_key_orders_higher_score_first():
    """No ties: a strictly higher score sorts before a strictly lower one."""
    high = _mk(0, score=0.95, phys=[10, 11, 12])
    low = _mk(1, score=0.90, phys=[1, 2, 3])
    placements = [low, high]
    placements.sort(key=_placement_sort_key)
    assert placements == [high, low]


def test_key_breaks_score_ties_by_physical_indices_ascending():
    """Two placements with identical scores: smaller physical_indices wins.

    This is the F5 acceptance bar — without this rule, ties resolve via
    Python's stable-sort over the vf2_mapping iteration order, an implicit
    undocumented ordering that a different lazy traversal cannot reproduce.
    """
    bigger_qubits = _mk(0, score=0.92, phys=[20, 21, 22])
    smaller_qubits = _mk(1, score=0.92, phys=[1, 2, 3])
    # Worst-case input order: ties present, smaller-qubits placement second.
    placements = [bigger_qubits, smaller_qubits]
    placements.sort(key=_placement_sort_key)
    assert placements == [smaller_qubits, bigger_qubits], (
        "F5 invariant: on score ties, smaller physical_indices must win."
    )


def test_key_is_idempotent():
    """Sorting a list twice yields the same order each time."""
    placements = [
        _mk(0, 0.85, [5, 6]),
        _mk(1, 0.95, [9, 10]),
        _mk(2, 0.85, [1, 2]),     # tied with id=0
        _mk(3, 0.90, [3, 4]),
        _mk(4, 0.95, [7, 8]),     # tied with id=1
    ]
    placements.sort(key=_placement_sort_key)
    first_order = [p.placement_id for p in placements]
    placements.sort(key=_placement_sort_key)
    second_order = [p.placement_id for p in placements]
    assert first_order == second_order


def test_key_is_deterministic_across_input_permutations():
    """Five randomized input orderings of the same set converge on one output.

    This is the structural test for "lazy traversal === eager enumerate" once
    the lazy path lands in W1.3: regardless of the order placements are
    visited, the sort key must produce a single canonical output sequence.
    """
    base = [
        _mk(0, 0.85, [5, 6]),
        _mk(1, 0.95, [9, 10]),
        _mk(2, 0.85, [1, 2]),     # tied with id=0
        _mk(3, 0.90, [3, 4]),
        _mk(4, 0.95, [7, 8]),     # tied with id=1
    ]
    rng = random.Random(0)
    orderings = []
    for _ in range(5):
        perm = list(base)
        rng.shuffle(perm)
        perm.sort(key=_placement_sort_key)
        orderings.append(tuple(p.placement_id for p in perm))
    assert len(set(orderings)) == 1, (
        f"F5 invariant: sort must be deterministic regardless of input order; "
        f"got distinct orderings {set(orderings)}"
    )


def test_key_handles_negative_scores():
    """Some scoring strategies (e.g. ``min_error``) return negative numbers;
    the ``-score`` negation must not flip the tie-break direction."""
    p1 = _mk(0, score=-0.10, phys=[5, 6])
    p2 = _mk(1, score=-0.10, phys=[1, 2])   # same negative score; smaller qubits
    p3 = _mk(2, score=-0.05, phys=[3, 4])   # better score than the other two
    placements = [p1, p2, p3]
    placements.sort(key=_placement_sort_key)
    assert placements == [p3, p2, p1], (
        "Negative-score handling: best (least-negative) first; tie still "
        "broken by physical_indices ascending."
    )


def test_top_1_selection_is_deterministic():
    """``top_1`` is the gate-2 device-calibrated guardrail; on score ties the
    selected placement must be reproducible (this is what the canary's
    ``QB11-QB5-QB6-QB7-QB13-QB21-QB29-QB28-QB27-QB26`` directory name in
    ``evidence/W1/gate2_canary/`` proves at the 2-seed scale)."""
    # Two placements with the SAME score, picked from a hypothetical Q50 q10
    # subgraph search: the tie-break must select the smaller-indexed set.
    pick_a = _mk(0, score=0.88, phys=[5, 6, 7, 11, 13, 21, 26, 27, 28, 29])
    pick_b = _mk(1, score=0.88, phys=[15, 16, 17, 21, 23, 31, 36, 37, 38, 39])
    for input_order in ([pick_a, pick_b], [pick_b, pick_a]):
        placements = list(input_order)
        placements.sort(key=_placement_sort_key)
        top_1 = placements[0]
        assert top_1 is pick_a, (
            f"top_1 must select the smaller-indexed placement on a tie; "
            f"got placement_id={top_1.placement_id} for input order "
            f"{[p.placement_id for p in input_order]}"
        )
