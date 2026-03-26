# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Abstract base class for ansatz circuit builders.

Each ansatz type (HVA, SU2, UCCSD, etc.) implements this interface.
The returned AnsatzMetadata drives downstream decisions: which gradient
strategy to use, which initializer, whether decomposition is needed.

To add a new ansatz:
  1. Create a new .py file in plugins/ansatze/
  2. Subclass AnsatzBuilder
  3. Implement build() — return (circuit, metadata)
  4. Set metadata.gradient_compatibility to declare what works
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

from lumi_hpc_qc.types import AnsatzMetadata, ExperimentConfig


class AnsatzBuilder(ABC):
    """Abstract builder for parameterized ansatz circuits."""

    name: str = ""

    @abstractmethod
    def build(
        self, num_qubits: int, config: ExperimentConfig
    ) -> tuple[Any, AnsatzMetadata]:
        """Build the parameterized ansatz circuit.

        Args:
            num_qubits: Number of qubits (from Hamiltonian)
            config: Experiment config (for ansatz_params like reps, entanglement)

        Returns:
            (circuit, metadata) where:
            - circuit is a QuantumCircuit with Parameter objects
            - metadata declares gradient_compatibility, preferred_initializer,
              requires_decomposition, parameter_names, etc.

        Contract:
            - len(circuit.parameters) == metadata.num_parameters
            - If metadata.requires_decomposition is True, the workflow
              must call backend.compile_circuit() before execution
            - metadata.gradient_compatibility must be one of:
              "parameter_shift", "finite_difference", "both"
        """
