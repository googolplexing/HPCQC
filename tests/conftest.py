# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Shared test fixtures for lumi-hpc-qc test suite.

Provides mock backends, small test Hamiltonians, and mock SLURM scheduler
so tests can run without GPU resources or SLURM access.
"""

from __future__ import annotations

import numpy as np
import pytest

from lumi_hpc_qc.types import (
    BackendCapabilities,
    CircuitJob,
    CircuitResult,
    ExperimentConfig,
    SlurmConfig,
)
from lumi_hpc_qc.backends.base import Backend


class MockBackend(Backend):
    """Test backend that returns a known energy without simulation.

    For a parameter vector θ, returns E = Σ(θ_i²) — a simple convex
    function with known minimum at θ=0, E=0. This lets us test the
    full VQE pipeline without needing Aer/GPU.
    """

    name = "mock"

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config
        self.call_count = 0

    def run_circuits(self, jobs: list[CircuitJob]) -> list[CircuitResult]:
        results = []
        for job in jobs:
            self.call_count += 1
            # Simple quadratic energy: E = sum(params^2) - offset
            if job.parameters:
                params = np.array(list(job.parameters[0].values()))
                energy = float(np.sum(params ** 2)) - 5.0
            else:
                energy = -5.0
            results.append(CircuitResult(
                job_id=job.job_id,
                energies=[energy],
                execution_time_s=0.001,
                backend_name="mock",
            ))
        return results

    def compile_circuit(self, circuit):
        return circuit  # no-op

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=100,
            supports_statevector=True,
            slurm_partition="standard",
        )

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        return []


@pytest.fixture
def mock_backend():
    """Provide a mock backend for testing."""
    return MockBackend()


@pytest.fixture
def sample_config():
    """Provide a minimal valid ExperimentConfig for testing."""
    return ExperimentConfig(
        model="fermi_hubbard",
        model_params={"lattice_rows": 2, "lattice_cols": 2, "hopping_t": 1.0, "interaction_u": 2.0},
        ansatz="hva",
        ansatz_params={"reps": 1},
        optimizer="l_bfgs_b",
        optimizer_params={"maxiter": 50},
        gradient="parameter_shift",
        initializer="random",
        initializer_params={"seed": 42},
        backend="aer_gpu",
        precision="double",
        num_qubits=8,
        mode="interactive",
        slurm=SlurmConfig(
            partition="standard-g",
            account="test_project",
            walltime="00:30:00",
            nodes=1,
            gpus_per_node=8,
        ),
        output_dir="/tmp/test_results",
    )


@pytest.fixture
def small_config():
    """Minimal 2-qubit config for fast integration tests."""
    return ExperimentConfig(
        model="byo",
        model_params={},
        ansatz="su2",
        ansatz_params={"reps": 1},
        optimizer="cobyla",
        optimizer_params={"maxiter": 20},
        gradient="none",
        initializer="random",
        initializer_params={"seed": 42},
        backend="mock",
        precision="double",
        num_qubits=2,
        mode="interactive",
        output_dir="/tmp/test_results",
    )
