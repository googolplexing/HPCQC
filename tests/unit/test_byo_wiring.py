# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Gap A wiring tests for byo_circuit (SPEC-002 §7.5) — run on LUMI in-container.

Exercises the engine path end to end on the build side: parse -> validate ->
expand (seed-outer/grid-inner) -> per-task build seam, plus the two fail-loud
guards (factory signature, default-ON cross-grid identity check). Needs qiskit
(the factory builds a real circuit) so it runs in the container, not on the Mac.

Relies on the committed example assets:
  examples/byo/floquet_dtc.py
  examples/byo/floquet_disorder_q4.json
  examples/q50_calibration_20260524_08c3c70f.json
Run from the repo root so the relative paths resolve.
"""
from __future__ import annotations

import pytest

from lumi_hpc_qc.sweep.sweep_engine import (
    parse_sweep_config,
    validate_sweep_config,
    expand_grid,
    SweepEngine,
)


def _cfg_dict():
    return {"sweep": {
        "experiments": [{
            "type": "byo_circuit",
            "label": "floquet_dtc_example",
            "circuit_script": "examples/byo/floquet_dtc.py",
            "circuit_function": "build_circuit",
            "fixed": {"num_qubits": 4, "epsilon": 0.03},
            "grid": {"num_kicks": {"range": [0, 6]}},   # stop-exclusive -> 0..5
            "disorder": {
                "source": "file",
                "file": "examples/byo/floquet_disorder_q4.json",
                "initial_state": 3,
            },
            "disorder_gates": ["rz", "rzz"],
            "seed_list": [0, 1, 2],
            "noise_configs": ["noiseless"],
        }],
        "calibrations": ["examples/q50_calibration_20260524_08c3c70f.json"],
    }}


def test_validate_clean():
    errs = validate_sweep_config(parse_sweep_config(_cfg_dict()))
    assert errs == [], errs


def test_expand_counts_order_and_params():
    tasks = expand_grid(parse_sweep_config(_cfg_dict()))
    # 1 calibration × 3 seeds × 6 grid points
    assert len(tasks) == 18, len(tasks)
    # seed is the OUTER axis, grid the INNER axis
    assert [t.seed for t in tasks[:6]] == [0] * 6
    assert [t.circuit_params["num_kicks"] for t in tasks[:6]] == [0, 1, 2, 3, 4, 5]
    assert [t.seed for t in tasks[6:12]] == [1] * 6
    for t in tasks:
        assert t.experiment_type == "byo_circuit"
        assert t.qubit_size == 4
        assert t.circuit_script.endswith("floquet_dtc.py")
        assert set(t.disorder_instance) == {"hz_angles", "Jzz_angles", "init_bit_array"}
        assert "num_kicks" in t.circuit_params
        # circuit_params stays separate from the Hamiltonian-routed model_params
        assert t.model_params == {}


def test_build_seam_realizes_circuit():
    cfg = parse_sweep_config(_cfg_dict())
    tasks = expand_grid(cfg)
    t5 = next(t for t in tasks if t.seed == 0 and t.circuit_params["num_kicks"] == 5)
    loaded = SweepEngine._build_byo_circuit(t5)
    assert loaded.num_qubits == 4
    assert loaded.gate_counts.get("rz", 0) == 5 * 4        # 5 kicks × 4 qubits
    assert loaded.gate_counts.get("rzz", 0) == 5 * 3       # 5 kicks × (4-1) edges
    # the t=0 reference builds with no disorder gates (the degenerate point the
    # cross-grid check must not trip on)
    t0 = next(t for t in tasks if t.seed == 0 and t.circuit_params["num_kicks"] == 0)
    assert SweepEngine._build_byo_circuit(t0).gate_counts.get("rz", 0) == 0


def test_signature_mismatch_raises():
    cfg = parse_sweep_config(_cfg_dict())
    cfg.experiments[0].fixed.pop("epsilon")     # remove a required factory param
    with pytest.raises(ValueError, match="requires"):
        expand_grid(cfg)


def test_cross_grid_catches_impure_factory(tmp_path):
    impure = tmp_path / "impure.py"
    impure.write_text(
        "import numpy as np\n"
        "from qiskit import QuantumCircuit\n"
        "def build_circuit(*, num_kicks, epsilon, num_qubits, hz_angles, Jzz_angles, init_bit_array):\n"
        "    qc = QuantumCircuit(num_qubits, num_qubits)\n"
        "    for _ in range(max(num_kicks, 1)):\n"
        "        for w in range(num_qubits):\n"
        "            qc.rz(np.random.uniform(-np.pi, np.pi), w)\n"   # ignores supplied hz_angles
        "        for w in range(num_qubits - 1):\n"
        "            qc.rzz(Jzz_angles[w], w, w + 1)\n"
        "    qc.measure(range(num_qubits), range(num_qubits))\n"
        "    return qc\n"
    )
    cfg = parse_sweep_config(_cfg_dict())
    cfg.experiments[0].circuit_script = str(impure)
    with pytest.raises(ValueError, match="FAILED"):
        expand_grid(cfg)


def _cfg_dict_with_placement():
    cfg = _cfg_dict()
    cfg["sweep"]["experiments"][0]["physical_qubits"] = ["QB1", "QB2", "QB5", "QB6"]
    return cfg


def test_expand_grid_propagates_physical_qubits_to_byo_tasks():
    # PLACEMENT-1 regression: experiment-level physical_qubits must reach EVERY
    # task, mirroring the shots/optimization_level carry. Without it the
    # executor reads None off the task and silently runs the solver -- the W1.6
    # Step-1 bug, which the prior dispatch test missed by using a fake task that
    # already had the attribute set. This drives the real expand_grid path.
    tasks = expand_grid(parse_sweep_config(_cfg_dict_with_placement()))
    assert tasks
    for t in tasks:
        assert t.physical_qubits == [["QB1", "QB2", "QB5", "QB6"]], t.physical_qubits


def test_expand_grid_no_physical_qubits_leaves_none():
    # Field-absent path stays None (solver self-selects) -- byte-identical to
    # pre-PLACEMENT-1 behaviour.
    tasks = expand_grid(parse_sweep_config(_cfg_dict()))
    assert tasks
    assert all(t.physical_qubits is None for t in tasks)
