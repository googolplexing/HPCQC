# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Unit tests for config loading and type validation."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import yaml

from lumi_hpc_qc.cli.config_loader import load_config
from lumi_hpc_qc.types import (
    ExperimentConfig,
    TimingBreakdown,
    _generate_experiment_id,
)


class TestExperimentId:
    def test_contains_uuid_and_slurm(self):
        eid = _generate_experiment_id()
        parts = eid.split("_")
        assert len(parts) == 2
        # First part is 12-char hex
        assert len(parts[0]) == 12
        # Second part is "interactive" (no SLURM in test env)
        assert parts[1] == "interactive"

    def test_unique(self):
        ids = {_generate_experiment_id() for _ in range(100)}
        assert len(ids) == 100


class TestConfigLoader:
    def test_minimal_config(self, tmp_path):
        config_data = {
            "model": "fermi_hubbard",
            "ansatz": "hva",
        }
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        assert config.model == "fermi_hubbard"
        assert config.ansatz == "hva"
        assert config.precision == "double"  # default
        assert config.optimizer == "l_bfgs_b"  # default

    def test_overrides(self, tmp_path):
        config_data = {"model": "fermi_hubbard", "ansatz": "hva", "precision": "double"}
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path, overrides={"precision": "single"})
        assert config.precision == "single"

    def test_missing_model_raises(self, tmp_path):
        config_data = {"ansatz": "hva"}
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="model.*required"):
            load_config(config_path)

    def test_invalid_precision_raises(self, tmp_path):
        config_data = {"model": "fermi_hubbard", "ansatz": "hva", "precision": "half"}
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        with pytest.raises(ValueError, match="precision"):
            load_config(config_path)

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_slurm_config_defaults(self, tmp_path):
        config_data = {"model": "fermi_hubbard", "ansatz": "hva"}
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        assert config.slurm.partition == "standard-g"
        assert config.slurm.nodes == 1

    def test_slurm_config_custom(self, tmp_path):
        config_data = {
            "model": "fermi_hubbard",
            "ansatz": "hva",
            "slurm": {
                "partition": "standard",
                "account": "project_123",
                "nodes": 4,
            },
        }
        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        config = load_config(config_path)
        assert config.slurm.partition == "standard"
        assert config.slurm.account == "project_123"
        assert config.slurm.nodes == 4


class TestTimingBreakdown:
    def test_human_readable(self):
        tb = TimingBreakdown(
            phases={"build": 2.5, "optimize": 10.0},
            total_s=12.5,
            percentages={"build": 20.0, "optimize": 80.0},
        )
        text = tb.to_human_readable()
        assert "build" in text
        assert "optimize" in text
        assert "TOTAL" in text

    def test_benchmark_json(self):
        tb = TimingBreakdown(
            phases={"build": 2.5},
            total_s=2.5,
            percentages={"build": 100.0},
        )
        d = tb.to_benchmark_json()
        assert d["phases"]["build"] == 2.5
        assert d["total_s"] == 2.5

    def test_training_record(self):
        tb = TimingBreakdown(
            phases={"hamiltonian build": 2.5, "vqe complete": 10.0},
            total_s=12.5,
            percentages={"hamiltonian build": 20.0, "vqe complete": 80.0},
        )
        rec = tb.to_training_record()
        assert "timing_hamiltonian_build_s" in rec
        assert "timing_vqe_complete_s" in rec
        assert rec["timing_total_s"] == 12.5
