# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Abstract base class for Hamiltonian builders.

Each physics model (Fermi-Hubbard, Heisenberg, etc.) implements this
interface. The workflow calls build() to get the qubit Hamiltonian
and uses the metadata to configure downstream components.

To add a new Hamiltonian:
  1. Create a new .py file in plugins/hamiltonians/
  2. Subclass HamiltonianBuilder
  3. Implement all abstract methods
  4. Set the `name` class attribute
  5. Plugin registry auto-discovers it
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

from lumi_hpc_qc.types import ExperimentConfig, HamiltonianMetadata


class HamiltonianBuilder(ABC):
    """Abstract builder for qubit Hamiltonians."""

    name: str = ""          # registry lookup key
    description: str = ""   # human-readable, written to experiment logs

    @abstractmethod
    def build(self, config: ExperimentConfig) -> tuple[Any, HamiltonianMetadata]:
        """Construct the qubit Hamiltonian from experiment config.

        Returns:
            (hamiltonian, metadata) where hamiltonian is a SparsePauliOp
            and metadata describes the operator for logging/validation.
        """

    @abstractmethod
    def exact_ground_energy(self, hamiltonian: Any) -> float | None:
        """Compute exact ground state energy via numpy diagonalization.

        Returns None if the Hamiltonian is too large for exact diag
        (typically >24 qubits on a single node).
        """

    @abstractmethod
    def adiabatic_parameter_name(self) -> str | None:
        """Name of the parameter ramped during adiabatic initialization.

        Returns:
            "interaction_u" for Fermi-Hubbard (U/t ratio)
            "jz" for Heisenberg (Jz coupling)
            None if adiabatic initialization is not applicable

        Used by AdiabaticInitializer to know which parameter to ramp
        from 0 to its target value.
        """

    @abstractmethod
    def build_at_parameter(
        self, value: float, config: ExperimentConfig
    ) -> Any:
        """Build Hamiltonian at a specific adiabatic parameter value.

        Used by AdiabaticInitializer during the ramp. For example,
        Fermi-Hubbard at U=0.5 builds the Hamiltonian with interaction_u=0.5
        while keeping all other parameters from config.

        Args:
            value: The adiabatic parameter value
            config: Full experiment config (for other parameters)

        Returns:
            SparsePauliOp at the specified parameter value
        """

    def default_params(self, num_qubits: int) -> dict[str, Any]:
        """Return default model_params for this Hamiltonian.

        Override in subclasses to provide plugin-specific defaults.
        The sweep engine calls this when no explicit model_params are
        provided in the YAML config (grid mode without LHS sampling).

        The base implementation returns {"num_qubits": num_qubits}.
        External plugins (e.g., DiagnosticTFIM) override this so that
        grid-mode sweeps produce correct typed parameter columns in
        Parquet without requiring a centralized switch statement.

        v1.2.1 — RED-DIRECTIVE-V121 Item 3.
        """
        return {"num_qubits": num_qubits}
