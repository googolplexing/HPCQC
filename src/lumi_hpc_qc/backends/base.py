# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Abstract base class for all execution backends.

This is the Dependency Inversion boundary between orchestration (Layer 2)
and execution (Layer 3). The workflow imports only this file — never
concrete backends like aer_gpu.py or iqm_qpu.py.

To add a new backend:
  1. Create a new file in backends/ (e.g., cudaq.py)
  2. Subclass Backend
  3. Implement all abstract methods
  4. The backend registry auto-discovers it
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

from lumi_hpc_qc.types import (
    BackendCapabilities,
    CircuitJob,
    CircuitResult,
    ExperimentConfig,
)


class Backend(ABC):
    """Abstract execution backend for quantum circuit evaluation.

    Backends handle the infrastructure concern: running circuits on
    GPUs, QPUs, or distributed resources. They know nothing about
    physics models, ansatze, or optimization — that's the workflow's job.
    """

    # Subclasses must set this — used by registry for lookup
    name: str = ""

    @abstractmethod
    def run_circuits(self, jobs: list[CircuitJob]) -> list[CircuitResult]:
        """Execute one or more circuit jobs and return results.

        Preconditions:
            - Circuits have been compiled for this backend via compile_circuit()
            - Parameter bindings in jobs match circuit parameters
        Postconditions:
            - Each CircuitJob maps 1:1 to a CircuitResult
            - Statevector jobs (shots=0) populate result.energies
            - Shot-based jobs populate result.counts
        Performance contract:
            - Backend MAY batch all circuits in a single simulator call
              for efficiency (e.g., Aer GPU batch submission)
        """

    @abstractmethod
    def compile_circuit(self, circuit: QuantumCircuit) -> QuantumCircuit:
        """Transpile / decompose a circuit for this backend's gate set.

        For Aer: multi-round decomposition to primitive gates
        For IQM: transpile to IQM native gate set
        For QCut: no-op (sub-backends compile their own fragments)
        """

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Declare what this backend can do.

        Used by:
            - Controller: to select the correct SLURM partition
            - Workflow: to validate that a circuit fits this backend
            - SLURM templates: to determine GPU vs CPU allocation
        """

    @abstractmethod
    def validate_config(self, config: ExperimentConfig) -> list[str]:
        """Validate that config is compatible with this backend.

        Returns:
            List of error messages. Empty list = config is valid.
        Checks may include:
            - Qubit count within backend limits
            - Precision mode supported
            - Required container / module available
        """

    def estimate_walltime(self, config: ExperimentConfig) -> int:
        """Estimate wall time in seconds for this experiment.

        Non-abstract: provides a conservative default based on qubit count.
        Backends should override with more accurate estimates based on
        their specific performance characteristics.
        """
        # Conservative default: 1 minute per qubit per 100 iterations
        n_qubits = config.num_qubits or 12
        max_iter = config.optimizer_params.get("maxiter", 200)
        return max(300, n_qubits * max_iter // 2)
