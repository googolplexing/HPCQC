"""PLACEMENT-1: shared placement-resolution seam + parse scope-gate.

Pure-Python tests. The ``resolve_placements`` dispatch is exercised with spy
solver primitives (no real rustworkx), and the parse scope-gate via
``parse_sweep_config`` (no circuit build) -- so this runs under the offline
harness and in-container alike.

The end-to-end experiment->task *propagation* test (which builds a real circuit
through ``expand_grid``) lives in ``test_byo_wiring.py``, the in-container BYO
expansion suite -- that is the check that would have caught the W1.6 Step-1
silent-solver bug, and it needs qiskit.
"""
from __future__ import annotations

import pytest

from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
from lumi_hpc_qc.sweep.sweep_engine import parse_sweep_config


# ── resolve_placements dispatch ─────────────────────────────────────────────
class _SpySolver(GeneralPlacementSolver):
    """Records which primitive the seam dispatched to, with exact kwargs."""

    def __init__(self):  # bypass real device registration
        self.calls: list[tuple[str, dict]] = []

    def find_all_placements(self, **kw):
        self.calls.append(("find_all_placements", kw))
        return ["SOLVER_RESULT"]

    def placements_from_names(self, **kw):
        self.calls.append(("placements_from_names", kw))
        return ["MANUAL_RESULT"]


_EDGES = [(0, 1), (1, 2), (2, 3)]


def test_resolve_dispatches_to_solver_when_no_manual():
    s = _SpySolver()
    out = s.resolve_placements(
        circuit_edges=_EDGES, circuit_qubits=4, device_id="Q50",
        strategy="max_fidelity", max_placements=7,
    )
    assert out == ["SOLVER_RESULT"]
    assert len(s.calls) == 1
    name, kw = s.calls[0]
    assert name == "find_all_placements"
    # Byte-identity of forwarded args vs the executors' historical direct call:
    # device_ids=[device_id], strategy, max_placements, default call_limit.
    assert kw == {
        "circuit_edges": _EDGES,
        "circuit_qubits": 4,
        "device_ids": ["Q50"],
        "strategy": "max_fidelity",
        "max_placements": 7,
        "call_limit": 100_000,
    }


def test_resolve_dispatches_to_manual_when_lists_given():
    s = _SpySolver()
    lists = [["QB1", "QB2", "QB5", "QB6"]]
    out = s.resolve_placements(
        circuit_edges=_EDGES, circuit_qubits=4, device_id="Q50",
        strategy="max_fidelity", max_placements=7,
        manual_qubit_name_lists=lists,
    )
    assert out == ["MANUAL_RESULT"]
    assert len(s.calls) == 1
    name, kw = s.calls[0]
    assert name == "placements_from_names"
    # Manual path forwards names/edges/qubits/device/strategy; it does NOT pass
    # max_placements or call_limit (placements_from_names returns exactly the
    # supplied placements). Matches the prior in-executor manual branch.
    assert kw == {
        "qubit_name_lists": lists,
        "circuit_edges": _EDGES,
        "circuit_qubits": 4,
        "device_id": "Q50",
        "strategy": "max_fidelity",
    }


def test_resolve_empty_manual_list_is_solver_path():
    # An empty list is falsy -> solver self-selects, matching the executors'
    # historical ``if manual_placements:`` truthiness.
    s = _SpySolver()
    s.resolve_placements(
        circuit_edges=_EDGES, circuit_qubits=4, device_id="Q50",
        manual_qubit_name_lists=[],
    )
    assert s.calls[0][0] == "find_all_placements"


# ── parse scope-gate ────────────────────────────────────────────────────────
_CAL = "examples/q50_calibration_20260524_08c3c70f.json"


def _byo_cfg(**extra):
    exp = {
        "type": "byo_circuit",
        "circuit_script": "examples/byo/floquet_dtc.py",
        "fixed": {"num_qubits": 4},
    }
    exp.update(extra)
    return {"sweep": {"experiments": [exp], "calibrations": [_CAL]}}


def _nonbyo_cfg(etype, **extra):
    exp = {"type": etype, "hamiltonians": ["tfim"], "qubit_sizes": [4]}
    exp.update(extra)
    return {"sweep": {"experiments": [exp], "calibrations": [_CAL]}}


def test_parse_allows_physical_qubits_on_byo():
    cfg = _byo_cfg(physical_qubits=["QB1", "QB2", "QB5", "QB6"])
    parsed = parse_sweep_config(cfg)
    # Single placement normalizes to a one-element list of lists, carried on the
    # experiment config.
    assert parsed.experiments[0].physical_qubits == [["QB1", "QB2", "QB5", "QB6"]]


def test_parse_rejects_physical_qubits_on_characterization():
    with pytest.raises(ValueError, match="byo_circuit"):
        parse_sweep_config(
            _nonbyo_cfg("characterization",
                        physical_qubits=["QB1", "QB2", "QB5", "QB6"])
        )


def test_parse_rejects_physical_qubits_on_vqe_sweep():
    with pytest.raises(ValueError, match="byo_circuit"):
        parse_sweep_config(
            _nonbyo_cfg("vqe_sweep",
                        physical_qubits=["QB1", "QB2", "QB5", "QB6"])
        )


def test_parse_nonbyo_without_physical_qubits_ok():
    # No false positive: a normal non-byo experiment still parses, field None.
    parsed = parse_sweep_config(_nonbyo_cfg("characterization"))
    assert parsed.experiments[0].physical_qubits is None
