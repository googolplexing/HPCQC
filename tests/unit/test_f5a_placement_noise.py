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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
