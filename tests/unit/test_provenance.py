# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Unit tests for data layer: provenance, checkpoint, result_store."""

from __future__ import annotations

import json

import numpy as np
import pytest

from lumi_hpc_qc.data.provenance import ProvenanceCollector
from lumi_hpc_qc.data.result_store import NumpyEncoder, load_json, save_json
from lumi_hpc_qc.orchestration.checkpoint import CheckpointManager


class TestProvenanceCollector:
    def test_capture_runs_without_error(self):
        collector = ProvenanceCollector()
        prov = collector.capture()
        assert prov.python_version  # should always be non-empty
        assert prov.cpu_model  # should always detect something

    def test_imported_modules_includes_numpy(self):
        collector = ProvenanceCollector()
        prov = collector.capture()
        assert "numpy" in prov.imported_modules

    def test_slurm_fields_empty_outside_slurm(self):
        collector = ProvenanceCollector()
        prov = collector.capture()
        # Outside SLURM, these should be empty strings
        assert prov.slurm_job_id == "" or prov.slurm_job_id.isdigit()


class TestCheckpointManager:
    def test_save_and_load_roundtrip(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        state = {
            "current_params": np.array([1.0, 2.0, 3.0]),
            "best_energy": -5.123,
            "iteration": 42,
        }
        path = mgr.save(state, iteration=42, experiment_id="test123")
        assert path.endswith(".json")

        loaded = mgr.load(path)
        assert loaded["best_energy"] == -5.123
        assert loaded["iteration"] == 42
        assert loaded["_checkpoint_meta"]["experiment_id"] == "test123"
        # numpy array round-tripped through JSON and restored by _restore_arrays
        assert isinstance(loaded["current_params"], np.ndarray)
        np.testing.assert_array_equal(loaded["current_params"], np.array([1.0, 2.0, 3.0]))

    def test_exists_finds_latest(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        mgr.save({"a": 1}, iteration=10, experiment_id="exp1")
        mgr.save({"a": 2}, iteration=20, experiment_id="exp1")
        mgr.save({"a": 3}, iteration=5, experiment_id="exp2")

        latest = mgr.exists("exp1")
        assert latest is not None
        assert "iter000020" in latest

    def test_exists_returns_none_for_missing(self, tmp_path):
        mgr = CheckpointManager(str(tmp_path))
        assert mgr.exists("nonexistent") is None


class TestResultStore:
    def test_numpy_roundtrip(self, tmp_path):
        path = tmp_path / "test.json"
        data = {
            "params": np.array([1.0, 2.0, 3.0]),
            "energy": np.float64(-1.5),
            "counts": {"00": 500, "11": 500},
        }
        save_json(data, path)
        loaded = load_json(path)

        assert isinstance(loaded["params"], np.ndarray)
        np.testing.assert_array_equal(loaded["params"], [1.0, 2.0, 3.0])
        assert loaded["energy"] == -1.5
        assert loaded["counts"]["00"] == 500

    def test_complex_number_roundtrip(self, tmp_path):
        path = tmp_path / "test.json"
        data = {"amplitude": complex(0.5, -0.3)}
        save_json(data, path)
        loaded = load_json(path)
        assert loaded["amplitude"] == complex(0.5, -0.3)
