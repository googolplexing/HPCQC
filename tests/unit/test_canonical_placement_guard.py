"""Guard test (RED ruling #2): pin the solver's deterministic canonical top_1
placement for the q10 Floquet chain on Q50.

The gated Option-1 placement is pinned by *name* via ``--physical-qubits`` on
both arms. This test is the **drift canary** for that name list: it asserts the
placement solver still emits the exact ordered top_1 the names were recorded
from, so a silent rustworkx/vf2 emission reorder (which would otherwise shift
the gated placement without warning) is caught here first. This is the F5
emission-order gap closed structurally for the gated placement rather than left
implicit on vf2 output.

In-container (needs rustworkx + the calibration), like the byo-wiring expand
tests — runs on LUMI with the unit suite, not under the offline harness.

`_CANONICAL` is RECORDED FROM THE SOLVER via extract_canonical_placement.py
(RED ruling #2: "recorded from the solver, not asserted"). Fill it from that
run's `canonical top_1` line before committing — until then
``test_canonical_top1_matches_recorded_order`` fails loudly by design.
"""
from __future__ import annotations

import json

from lumi_hpc_qc.plugins.registry import PluginRegistry
from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver

_CAL = "examples/q50_calibration_20260524_08c3c70f.json"
_CHAIN = [(i, i + 1) for i in range(9)]   # q10 open linear Floquet chain (9 edges)

# Recorded from extract_canonical_placement.py — FILL before committing, e.g.:
#   _CANONICAL = ["QB11", "QB5", "QB6", "QB7", "QB13", "QB21", "QB29", "QB28", "QB27", "QB26"]
_CANONICAL: list[str] | None = None


def _top1_order() -> tuple[list[str], object]:
    """Solver's deterministic top_1 placement order (logical 0..9 -> physical),
    built exactly as extract_canonical_placement.py and the engine build it."""
    cal_json = json.load(open(_CAL))
    reg = PluginRegistry()
    reg.discover()                 # engine does this at construction (sweep_engine.py:1313)
    adapter = reg.get_calibration_adapter(cal_json.get("adapter", "iqm_v2"))
    device_cal = adapter.load(_CAL)
    solver = GeneralPlacementSolver()
    solver.add_device(device_cal)
    pls = solver.find_all_placements(
        circuit_edges=_CHAIN, circuit_qubits=10,
        device_ids=[device_cal.device_id],
        strategy="max_fidelity", max_placements=1,
    )
    assert pls, "no placements returned for the q10 chain on Q50"
    p = pls[0]
    return [p.qubit_mapping[i] for i in range(10)], device_cal


def test_canonical_top1_matches_recorded_order():
    # The drift canary: the gated placement is pinned by name; this ensures the
    # solver still emits that exact ordered top_1.
    assert _CANONICAL is not None, (
        "fill _CANONICAL from extract_canonical_placement.py "
        "('canonical top_1' line) before committing this guard test"
    )
    order, _ = _top1_order()
    assert order == _CANONICAL, f"solver top_1 order drifted: {order} != {_CANONICAL}"


def test_canonical_top1_is_deterministic():
    # Two independent solves must agree — guards against any nondeterminism in
    # the emission that a single-shot record could mask.
    a, _ = _top1_order()
    b, _ = _top1_order()
    assert a == b, f"solver top_1 nondeterministic across calls: {a} != {b}"


def test_canonical_top1_is_a_valid_chain():
    # Consecutive qubits in the top_1 order must be real calibrated couplers,
    # i.e. the placement is a Hamiltonian path the linear circuit sits on with
    # no routing — the precondition that lets --physical-qubits pin it directly.
    order, _ = _top1_order()
    assert len(order) == 10 and len(set(order)) == 10, order
    cal = json.load(open(_CAL))
    adj: dict[str, set[str]] = {}
    for gate_pair in cal["two_qubit_gates"]:
        a, b = gate_pair.split("-")
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    for i in range(9):
        assert order[i + 1] in adj.get(order[i], set()), (
            f"top_1 edge {order[i]}-{order[i + 1]} is not a calibrated coupler"
        )
