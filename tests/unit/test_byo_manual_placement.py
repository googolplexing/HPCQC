# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""PLACEMENT-1 — researcher-controlled placement selection.

Covers the two Phase-1 pieces (BLUE-PROPOSAL-RESEARCHER-PLACEMENT-CONTROL):

  (1) _parse_physical_qubits: the YAML normalizer — a single placement
      (list of qubit strings) is wrapped to one element; several placements
      (list of lists) pass through; malformed values fail loud.

  (2) PlacementSolver.placements_from_names: builds faithful Placement objects
      from explicit qubit-name lists, bypassing the subgraph solver, with
      fail-loud validation (count, duplicates, names-in-calibration, real
      device edges) and logical-order qubit_mapping (logical i -> names[i]).

The solver's metric helpers are pure calibration-dict based (no rustworkx), so
a minimal fake calibration exercises the full construction path offline.
"""

from __future__ import annotations

import pytest

from lumi_hpc_qc.sweep.sweep_engine import _parse_physical_qubits
from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver


# ── (1) _parse_physical_qubits normalizer ──────────────────────────────────

def test_parse_none_returns_none():
    assert _parse_physical_qubits(None) is None


def test_parse_single_placement_is_wrapped():
    # a bare list of names == one placement
    assert _parse_physical_qubits(["QB1", "QB2", "QB3"]) == [["QB1", "QB2", "QB3"]]


def test_parse_multiple_placements_pass_through():
    raw = [["QB1", "QB2"], ["QB3", "QB4"]]
    assert _parse_physical_qubits(raw) == [["QB1", "QB2"], ["QB3", "QB4"]]


def test_parse_rejects_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _parse_physical_qubits([])


def test_parse_rejects_mixed_str_and_list():
    with pytest.raises(ValueError, match="not a mix"):
        _parse_physical_qubits(["QB1", ["QB2", "QB3"]])


def test_parse_rejects_non_string_qubit():
    with pytest.raises(ValueError, match="qubit-name strings"):
        _parse_physical_qubits([["QB1", 7]])


# ── (2) PlacementSolver.placements_from_names ──────────────────────────────

class _FakeQubit:
    def __init__(self, t1, t2, ro_fid, ge):
        self.t1_us = t1
        self.t2_us = t2
        self.readout_fidelity = ro_fid
        self.single_gate_error = ge


class _FakeCal:
    """Minimal DeviceCalibration stand-in: 5 qubits in a chain 0-1-2-3-4."""
    device_id = "vtt_q50_test"
    device_prefix = "vtt_q50"
    num_qubits = 5

    def __init__(self):
        names = [f"QB{i}" for i in range(5)]
        self.index_to_qubit_name = {i: n for i, n in enumerate(names)}
        self.qubits = {n: _FakeQubit(30.0, 12.0, 0.97, 0.0005) for n in names}
        # chain adjacency
        self.adjacency = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}

    def gate_fidelity(self, i, j):
        return 0.99 if j in self.adjacency.get(i, set()) else 0.0


def _solver():
    s = GeneralPlacementSolver()
    s._devices["vtt_q50_test"] = _FakeCal()
    return s


# a 3q line circuit: edges (0,1),(1,2)
LINE3_EDGES = [(0, 1), (1, 2)]


def test_single_placement_builds_faithful_placement():
    s = _solver()
    pls = s.placements_from_names(
        qubit_name_lists=[["QB1", "QB2", "QB3"]],
        circuit_edges=LINE3_EDGES,
        circuit_qubits=3,
        device_id="vtt_q50_test",
    )
    assert len(pls) == 1
    p = pls[0]
    # logical order preserved: logical i -> names[i] (the F5a convention)
    assert p.qubit_mapping == {0: "QB1", 1: "QB2", 2: "QB3"}
    # physical_indices canonical (sorted ascending), like the solver
    assert p.physical_indices == [1, 2, 3]
    assert p.device_id == "vtt_q50_test"
    assert p.device_prefix == "vtt_q50"
    # faithful metric fields are populated (not left at defaults)
    assert p.internal_edges == 2          # QB1-QB2 and QB2-QB3 are real edges
    assert p.avg_readout_fidelity > 0
    assert p.topology_hash != ""
    assert set(p.per_qubit_calibration) == {"QB1", "QB2", "QB3"}


def test_multiple_placements_preserve_input_order_and_ids():
    s = _solver()
    pls = s.placements_from_names(
        qubit_name_lists=[["QB0", "QB1", "QB2"], ["QB2", "QB3", "QB4"]],
        circuit_edges=LINE3_EDGES,
        circuit_qubits=3,
        device_id="vtt_q50_test",
    )
    assert len(pls) == 2
    assert pls[0].qubit_mapping[0] == "QB0"
    assert pls[1].qubit_mapping[0] == "QB2"
    assert [p.placement_id for p in pls] == [0, 1]


def test_wrong_qubit_count_fails():
    s = _solver()
    with pytest.raises(ValueError, match="must match"):
        s.placements_from_names(
            qubit_name_lists=[["QB1", "QB2"]],          # 2 != 3
            circuit_edges=LINE3_EDGES, circuit_qubits=3,
            device_id="vtt_q50_test",
        )


def test_unknown_qubit_name_fails():
    s = _solver()
    with pytest.raises(ValueError, match="not in calibration"):
        s.placements_from_names(
            qubit_name_lists=[["QB1", "QB2", "QB99"]],   # QB99 absent
            circuit_edges=LINE3_EDGES, circuit_qubits=3,
            device_id="vtt_q50_test",
        )


def test_duplicate_qubit_fails():
    s = _solver()
    with pytest.raises(ValueError, match="repeats"):
        s.placements_from_names(
            qubit_name_lists=[["QB1", "QB1", "QB2"]],
            circuit_edges=LINE3_EDGES, circuit_qubits=3,
            device_id="vtt_q50_test",
        )


def test_non_adjacent_edge_fails():
    s = _solver()
    # QB0 and QB2 are NOT adjacent (chain is 0-1-2-3-4), so a circuit edge
    # mapping logical (0,1) -> (QB0, QB2) is not a real 2q gate.
    with pytest.raises(ValueError, match="not a\\s+calibrated 2q gate"):
        s.placements_from_names(
            qubit_name_lists=[["QB0", "QB2", "QB4"]],
            circuit_edges=[(0, 1)],                      # QB0-QB2: no edge
            circuit_qubits=3,
            device_id="vtt_q50_test",
        )


def test_unregistered_device_fails():
    s = _solver()
    with pytest.raises(ValueError, match="not registered"):
        s.placements_from_names(
            qubit_name_lists=[["QB1", "QB2", "QB3"]],
            circuit_edges=LINE3_EDGES, circuit_qubits=3,
            device_id="nonexistent",
        )


# ── (3) RED §6 condition 2 — list-wise validation + cross-calibration drift ──
# RED-RESP-GATE2 §6(2): fail-loud must be tested by assertion, list-wise:
# per-placement length, per-placement name resolution, and the drift case
# (a name valid in one calibration, absent in another) — never a silent
# fall-back to the solver.

class _OldFakeCal:
    """An *older* calibration era: only 3 qubits (QB0,QB1,QB2), chain 0-1-2.

    Models the replication-on-older-cal case: a placement chosen against a
    newer 5-qubit snapshot references a qubit (QB3/QB4) absent from this era.
    """
    device_id = "vtt_q50_old"
    device_prefix = "vtt_q50"
    num_qubits = 3

    def __init__(self):
        names = [f"QB{i}" for i in range(3)]
        self.index_to_qubit_name = {i: n for i, n in enumerate(names)}
        self.qubits = {n: _FakeQubit(30.0, 12.0, 0.97, 0.0005) for n in names}
        self.adjacency = {0: {1}, 1: {0, 2}, 2: {1}}

    def gate_fidelity(self, i, j):
        return 0.99 if j in self.adjacency.get(i, set()) else 0.0


def _solver_two_eras():
    """Solver with both a current (5q) and an older (3q) calibration registered."""
    s = GeneralPlacementSolver()
    s._devices["vtt_q50_test"] = _FakeCal()   # current: QB0..QB4
    s._devices["vtt_q50_old"] = _OldFakeCal()  # older:   QB0..QB2
    return s


def test_listwise_one_placement_wrong_length_fails():
    # Several placements; only the SECOND has the wrong length. Must raise
    # (not silently drop it, not fall back to the solver).
    s = _solver()
    with pytest.raises(ValueError, match="must match"):
        s.placements_from_names(
            qubit_name_lists=[["QB0", "QB1", "QB2"], ["QB3", "QB4"]],  # 2nd: 2!=3
            circuit_edges=LINE3_EDGES, circuit_qubits=3,
            device_id="vtt_q50_test",
        )


def test_listwise_one_placement_unknown_name_fails():
    # Several placements; only the SECOND names an absent qubit. Must raise.
    s = _solver()
    with pytest.raises(ValueError, match="not in calibration"):
        s.placements_from_names(
            qubit_name_lists=[["QB0", "QB1", "QB2"], ["QB1", "QB2", "QB99"]],
            circuit_edges=LINE3_EDGES, circuit_qubits=3,
            device_id="vtt_q50_test",
        )


def test_drift_same_names_valid_in_one_cal_absent_in_older_cal():
    # The exact replication failure mode: a placement valid against the
    # current snapshot must FAIL-LOUD against an older calibration missing
    # one of its qubits — never silently mis-key or fall back to the solver.
    s = _solver_two_eras()
    names = [["QB1", "QB2", "QB3"]]  # QB3 exists now, not in the older era

    # valid against the current snapshot
    pls = s.placements_from_names(
        qubit_name_lists=names, circuit_edges=LINE3_EDGES,
        circuit_qubits=3, device_id="vtt_q50_test",
    )
    assert len(pls) == 1 and pls[0].qubit_mapping == {0: "QB1", 1: "QB2", 2: "QB3"}

    # absent in the older era -> raises, naming the offending qubit + device
    with pytest.raises(ValueError, match="not in calibration .*vtt_q50_old"):
        s.placements_from_names(
            qubit_name_lists=names, circuit_edges=LINE3_EDGES,
            circuit_qubits=3, device_id="vtt_q50_old",
        )


# ── (4) RED §6 condition 1 — field-absent dispatch is byte-identical ─────────
# RED-RESP-GATE2 §6(1): a no-physical_qubits case must take the SAME solver
# path with the SAME arguments as before the seam (and must NOT touch the new
# manual path); the present case must route to the manual path and NOT call
# the solver. Tested at the dispatch in _execute_byo_group with a spy solver
# and faked circuit-build, so the real branch logic runs offline.

from collections import defaultdict
from lumi_hpc_qc.sweep.sweep_engine import SweepEngine


class _FakeLoaded:
    def __init__(self):
        self.num_qubits = 3
        self.connectivity = [(0, 1), (1, 2)]


class _FakeDeviceCal:
    device_id = "vtt_q50_test"


class _FakeTask:
    def __init__(self, *, physical_qubits):
        self.calibration_path = "cal/path.json"
        self.circuit_params = {"num_kicks": 4}
        self.circuit_script = "examples/byo/floquet_dtc.py"
        self.noise_configs = []          # noiseless -> wants_device_cal = False
        self.physical_qubits = physical_qubits
        self.max_placements = 7          # distinctive: must be forwarded verbatim


class _SpySolver:
    def __init__(self):
        self.find_all_kwargs = None
        self.from_names_kwargs = None

    def find_all_placements(self, **kwargs):
        self.find_all_kwargs = kwargs
        return []                        # -> method exits at `if not placements`

    def placements_from_names(self, **kwargs):
        self.from_names_kwargs = kwargs
        return []


def _engine_with(spy, task):
    eng = object.__new__(SweepEngine)            # bypass heavy __init__
    eng._cal_cache = {task.calibration_path: ("cal_id", {}, _FakeDeviceCal())}
    eng._timing = defaultdict(float)
    eng._solver = spy
    eng._build_byo_circuit = lambda _t: _FakeLoaded()
    return eng


def test_field_absent_dispatches_to_solver_with_verbatim_args():
    spy = _SpySolver()
    task = _FakeTask(physical_qubits=None)       # field absent
    eng = _engine_with(spy, task)
    eng._execute_byo_group([task], writer=None, errors=[])

    # the manual path is NOT touched ...
    assert spy.from_names_kwargs is None
    # ... and the solver is called with exactly the pre-seam arguments.
    assert spy.find_all_kwargs == {
        "circuit_edges": [(0, 1), (1, 2)],
        "circuit_qubits": 3,
        "device_ids": ["vtt_q50_test"],
        "strategy": "max_fidelity",
        "max_placements": 7,
    }


def test_field_present_bypasses_solver():
    spy = _SpySolver()
    task = _FakeTask(physical_qubits=[["QB1", "QB2", "QB3"]])
    eng = _engine_with(spy, task)
    eng._execute_byo_group([task], writer=None, errors=[])

    # routes to the manual path with the supplied list, in logical order ...
    assert spy.from_names_kwargs == {
        "qubit_name_lists": [["QB1", "QB2", "QB3"]],
        "circuit_edges": [(0, 1), (1, 2)],
        "circuit_qubits": 3,
        "device_id": "vtt_q50_test",
    }
    # ... and the solver self-selection is NOT invoked.
    assert spy.find_all_kwargs is None
