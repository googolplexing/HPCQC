# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Abstract base class for parameter initialization strategies.

Different ansatze need different initialization: UCCSD needs zero (HF state),
strongly correlated models benefit from adiabatic ramping, generic ansatze
use random initialization. The ansatz metadata declares preferred_initializer
and the workflow selects accordingly.

To add a new initializer:
  1. Create a new .py file in plugins/initializers/
  2. Subclass InitializerStrategy
  3. Implement initialize()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from qiskit.circuit import QuantumCircuit

    from lumi_hpc_qc.backends.base import Backend
    from lumi_hpc_qc.plugins.hamiltonians.base import HamiltonianBuilder

from lumi_hpc_qc.types import ExperimentConfig


class InitializerStrategy(ABC):
    """Abstract parameter initialization strategy."""

    name: str = ""

    @abstractmethod
    def initialize(
        self,
        num_params: int,
        hamiltonian_builder: HamiltonianBuilder | None = None,
        ansatz: QuantumCircuit | None = None,
        backend: Backend | None = None,
        config: ExperimentConfig | None = None,
    ) -> np.ndarray:
        """Generate initial parameter vector.

        Different initializers use different subsets of these arguments:

        RandomInit:
            Only needs num_params + config (for seed and range).
        ZeroInit:
            Only needs num_params. Returns np.zeros(num_params).
            Used for UCCSD where θ=0 is the Hartree-Fock reference.
        AdiabaticInit:
            Needs ALL arguments. Builds intermediate Hamiltonians via
            hamiltonian_builder.build_at_parameter(), runs mini-optimizations
            at each ramp step using the provided backend for circuit evaluation.

        Args:
            num_params: Number of variational parameters in the ansatz.
            hamiltonian_builder: For building intermediate Hamiltonians
                during adiabatic ramp.
            ansatz: The parameterized circuit (needed by adiabatic init
                to evaluate energies at each ramp step).
            backend: Execution backend for circuit evaluation during
                adiabatic initialization.
            config: Full experiment config for initializer-specific
                parameters (seed, range, adiabatic_steps, etc.)

        Returns:
            Parameter array of shape (num_params,).
        """
