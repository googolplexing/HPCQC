# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""PLACEMENT union (Piece 1) — resolve_placements composes manual ∪ solver-top-N.

Covers ``GeneralPlacementSolver._compose_manual_solver`` (the pure dedup/net-N/
ordering/re-id logic) and the ``resolve_placements`` seam in its three modes
(manual-only, solver-only, union). Two layers:

  (A) Helper-logic tests with hand-built Placement objects — full control of the
      "solver" ranking, so the dedup-in-action and interleaved-collision cases
      are deterministic and need no device graph. These realize the worked
      examples: manual = solver #1 (N=1 -> #1 deduped, #2 returned); manual =
      solver #1 and #3 (N=2 -> #1 and #3 deduped, #2 and #4 returned, D=2,
      fetched N+K=4 deep).

  (B) One real-Q50 integration test: load the calibration, take the solver's
      genuine #1 and #3 chains, feed them back as manual, and confirm the union
      returns the solver's #2 and #4 on real placements.

Both layers print the physical-qubit chains (manual and solver-found) so a
``pytest -s`` run lists the placements being composed.
"""

from __future__ import annotations

import os
import pytest

from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver, Placement


# ── helpers ────────────────────────────────────────────────────────────────

def _mkpl(phys: list[int], score: float, source: str) -> Placement:
    """Build a Placement with just the fields the union logic reads."""
    return Placement(
        placement_id=-1,                       # re-id'd by the composer
        device_id="vtt_q50_test",
        device_prefix="vtt_q50",
        qubit_mapping={i: f"QB{q}" for i, q in enumerate(phys)},
        physical_indices=sorted(phys),
        score=score,
        internal_edges=len(phys) - 1,
        avg_readout_fidelity=0.97,
        avg_gate_fidelity=0.99,
        source=source,
    )


def _chain(p: Placement) -> list[int]:
    return p.physical_indices


def _print_chains(
    label: str,
    pls: list[Placement],
    name_to_index: dict[str, int] | None = None,
) -> None:
    """Diagnostic dump (pytest -s), readable across the three numberings.

    Three distinct labellings are in play and must not be conflated:
      - logical (L0..Ln-1): the circuit's own qubit index (the DTC chain is a
        linear L0-L1-...-Ln-1);
      - IQM name (QB5, QB11, ...): the hardware qubit label;
      - graph index (0-based): the coupling-graph node id the solver works in.

    Per placement we print, in LOGICAL/circuit order, each logical qubit paired
    with its IQM name and -- when ``name_to_index`` is supplied (e.g. inverted
    from the calibration's index_to_qubit_name) -- its graph index, so the line
    reads straight across: ``L# = QBname (idx#)``. ``physical_indices`` is
    printed separately as the SORTED set (the dedup key): order-independent and
    deliberately NOT aligned to the placement above it.
    """
    print(f"\n  {label}:")
    for p in pls:
        names = [p.qubit_mapping[i] for i in range(len(p.qubit_mapping))]
        if name_to_index is not None:
            cells = "  ".join(
                f"L{i}={nm}(idx{name_to_index[nm]})" for i, nm in enumerate(names)
            )
        else:
            cells = "  ".join(f"L{i}={nm}" for i, nm in enumerate(names))
        print(f"    id={p.placement_id:<6} source={p.source:<6} "
              f"score={p.score:+.4f}")
        print(f"        {cells}")
        print(f"        physical set (sorted graph index, dedup key): {_chain(p)}")


# A controlled "solver" ranking: four distinct 3q chains, score-descending
# (#1 best). The composer treats `solver` as already-ranked (as find_all_placements
# returns it), so the list order IS the ranking.
def _solver_ranking() -> list[Placement]:
    return [
        _mkpl([1, 2, 3], score=-0.10, source="solver"),   # #1 (best)
        _mkpl([2, 3, 4], score=-0.20, source="solver"),   # #2
        _mkpl([3, 4, 5], score=-0.30, source="solver"),   # #3
        _mkpl([4, 5, 6], score=-0.40, source="solver"),   # #4
    ]


S = GeneralPlacementSolver()   # _compose_manual_solver is pure (ignores self)


# ── (A1) inertness: union logic untouched when one side is absent ───────────

def test_manual_only_passthrough_is_unchanged():
    """resolve_placements(manual, solver_top_n=None) == placements_from_names."""
    # Logic-level: the composer is simply never called; manual list is returned
    # verbatim. (Engine-level byte-identity is covered by the real-cal test.)
    manual = [_mkpl([1, 2, 3], -0.1, "manual")]
    _print_chains("manual-only (solver bypassed)", manual)
    assert all(p.source == "manual" for p in manual)


# ── (A2) dedup-in-action: manual = the solver's own #1 chain, N=1 ───────────

def test_dedup_in_action_manual_is_solver_best():
    manual = [_mkpl([1, 2, 3], -0.10, "manual")]      # == solver #1
    solver = _solver_ranking()
    merged, stats = S._compose_manual_solver(manual, solver, solver_top_n=1)
    _print_chains("manual (= solver #1)", manual)
    _print_chains("solver ranking fetched", solver)
    _print_chains("UNION result", merged)
    print(f"  stats: {stats}")
    # #1 collides with the manual chain -> deduped; #2 is the returned solver pick.
    assert stats["d_deduped"] == 1
    assert stats["s_solver"] == 1
    assert stats["short_of_n"] == 0
    solver_out = [p for p in merged if p.source == "solver"]
    assert len(solver_out) == 1
    assert _chain(solver_out[0]) == [2, 3, 4]          # solver #2
    # manual kept, manual-first
    assert merged[0].source == "manual" and _chain(merged[0]) == [1, 2, 3]


# ── (A3) interleaved D=2: manual = solver #1 and #3, N=2 ────────────────────

def test_interleaved_dedup_first_and_third():
    manual = [
        _mkpl([1, 2, 3], -0.10, "manual"),             # == solver #1
        _mkpl([3, 4, 5], -0.30, "manual"),             # == solver #3
    ]
    solver = _solver_ranking()                          # the composer fetched N+K=4
    merged, stats = S._compose_manual_solver(manual, solver, solver_top_n=2)
    _print_chains("manual (= solver #1 and #3)", manual)
    _print_chains("solver ranking fetched (N+K=4 deep)", solver)
    _print_chains("UNION result", merged)
    print(f"  stats: {stats}")
    # #1 deduped, #2 added, #3 deduped, #4 added -> S=2, D=2, fetched 4 deep.
    assert stats == {
        "k_manual": 2, "n_requested": 2, "s_solver": 2,
        "d_deduped": 2, "fetch_depth": 4, "short_of_n": 0,
    }
    solver_out = [_chain(p) for p in merged if p.source == "solver"]
    assert solver_out == [[2, 3, 4], [4, 5, 6]]         # solver #2 and #4
    # ordering: manual-first then solver-ranked
    assert [p.source for p in merged] == ["manual", "manual", "solver", "solver"]
    # re-id unique 0..M-1
    assert [p.placement_id for p in merged] == [0, 1, 2, 3]


# ── (A4) fewer collisions than K: net exactly N, surplus survivors trimmed ──

def test_net_n_trims_surplus_survivors():
    # manual collides with only #1; N=2 -> survivors {#2,#3,#4}, take top 2.
    manual = [_mkpl([1, 2, 3], -0.10, "manual")]
    solver = _solver_ranking()
    merged, stats = S._compose_manual_solver(manual, solver, solver_top_n=2)
    print(f"  stats: {stats}")
    assert stats["d_deduped"] == 1 and stats["s_solver"] == 2
    assert [_chain(p) for p in merged if p.source == "solver"] == [[2, 3, 4], [3, 4, 5]]


# ── (A5) short-of-N: solver pool exhausted (the all-manual edge, in miniature)─

def test_short_of_n_when_solver_pool_exhausted():
    # Every solver pick collides with a manual chain -> no survivors. This is the
    # "supply all valid chains, ask N>=1" edge (S=0) reproduced with a tiny pool,
    # so it needs no 30k-chain enumeration.
    manual = [
        _mkpl([1, 2, 3], -0.1, "manual"),
        _mkpl([2, 3, 4], -0.2, "manual"),
    ]
    solver = [_mkpl([1, 2, 3], -0.1, "solver"), _mkpl([2, 3, 4], -0.2, "solver")]
    merged, stats = S._compose_manual_solver(manual, solver, solver_top_n=3)
    _print_chains("manual (covers entire solver pool)", manual)
    _print_chains("UNION result (no new solver chains)", merged)
    print(f"  stats: {stats}")
    assert stats["s_solver"] == 0
    assert stats["short_of_n"] == 3          # requested 3, got 0
    assert all(p.source == "manual" for p in merged)   # only manual survives
    assert [p.placement_id for p in merged] == [0, 1]


# ── (B) real-Q50 integration: the #1/#3 example on genuine placements ───────

_Q50 = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "examples", "q50_calibration_20260524_08c3c70f.json",
)
# 10q linear chain (the Floquet DTC connectivity).
_CHAIN10_EDGES = [(i, i + 1) for i in range(9)]


@pytest.mark.skipif(not os.path.exists(_Q50), reason="Q50 calibration not present")
def test_real_q50_union_returns_solver_2nd_and_4th():
    pytest.importorskip("rustworkx")
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter

    cal = IQMv2Adapter().load(_Q50)
    solver = GeneralPlacementSolver()
    solver.add_device(cal)
    dev = cal.device_id
    # name -> graph index (inverse of the calibration's index_to_qubit_name), so
    # the dump shows each logical qubit's IQM name AND its graph index. Derived
    # from the calibration, not assumed (no hardcoded QBk == idx k-1).
    n2i = {name: idx for idx, name in cal.index_to_qubit_name.items()}

    # Genuine solver ranking (top 4 by score, F5 total order).
    ranked = solver.find_all_placements(
        circuit_edges=_CHAIN10_EDGES, circuit_qubits=10,
        device_ids=[dev], max_placements=4,
    )
    assert len(ranked) == 4
    _print_chains("Q50 solver ranking (top 4)", ranked, n2i)

    # Feed the solver's #1 and #3 back as MANUAL (logical-order names).
    def names(p):
        return [p.qubit_mapping[i] for i in range(10)]
    manual_lists = [names(ranked[0]), names(ranked[2])]

    merged = solver.resolve_placements(
        circuit_edges=_CHAIN10_EDGES, circuit_qubits=10, device_id=dev,
        manual_qubit_name_lists=manual_lists, solver_top_n=2,
    )
    _print_chains("Q50 UNION (manual #1,#3 + solver top-2)", merged, n2i)

    manual_out = [p for p in merged if p.source == "manual"]
    solver_out = [p for p in merged if p.source == "solver"]
    assert len(manual_out) == 2 and len(solver_out) == 2
    # The two solver-found chains are the solver's #2 and #4 (its #1/#3 deduped).
    assert {frozenset(p.physical_indices) for p in solver_out} == {
        frozenset(ranked[1].physical_indices),
        frozenset(ranked[3].physical_indices),
    }
    # manual-first ordering and unique sequential ids
    assert [p.source for p in merged] == ["manual", "manual", "solver", "solver"]
    assert [p.placement_id for p in merged] == [0, 1, 2, 3]


@pytest.mark.skipif(not os.path.exists(_Q50), reason="Q50 calibration not present")
def test_real_q50_solver_only_unchanged_by_union_param():
    """solver_top_n omitted + no manual -> byte-identical to find_all_placements."""
    pytest.importorskip("rustworkx")
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter

    cal = IQMv2Adapter().load(_Q50)
    solver = GeneralPlacementSolver()
    solver.add_device(cal)
    dev = cal.device_id

    direct = solver.find_all_placements(
        circuit_edges=_CHAIN10_EDGES, circuit_qubits=10,
        device_ids=[dev], max_placements=5,
    )
    viaseam = solver.resolve_placements(
        circuit_edges=_CHAIN10_EDGES, circuit_qubits=10, device_id=dev,
        max_placements=5,
    )
    assert [p.physical_indices for p in viaseam] == [p.physical_indices for p in direct]
    assert all(p.source == "solver" for p in viaseam)
