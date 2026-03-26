# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Abstract base class for gradient computation strategies.

The workflow selects the gradient strategy based on ansatz metadata:
if the ansatz declares gradient_compatibility="parameter_shift", the
parameter-shift rule is used. Otherwise, finite-difference is the
universal fallback.

To add a new gradient method:
  1. Create a new .py file in plugins/gradients/
  2. Subclass GradientStrategy
  3. Implement compute() and validate_ansatz()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from lumi_hpc_qc.backends.base import Backend

from lumi_hpc_qc.types import AnsatzMetadata


class GradientStrategy(ABC):
    """Abstract gradient computation strategy."""

    name: str = ""

    @property
    @abstractmethod
    def circuits_per_gradient(self) -> str:
        """Human-readable description of circuit cost.

        Examples: "2n" for parameter-shift, "n+1" for forward finite-diff.
        Used in logging output.
        """

    @abstractmethod
    def compute(
        self,
        eval_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
        backend: Backend | None = None,
    ) -> np.ndarray:
        """Compute gradient vector at the given parameters.

        Args:
            eval_fn: Evaluates energy E(θ). Calls backend internally.
            params: Current parameter vector, shape (n_params,).
            backend: Optional backend reference for future batched
                gradient implementations where all shifted circuits
                are submitted in a single sim.run() call.

        Returns:
            Gradient array of shape (n_params,).
        """

    @abstractmethod
    def validate_ansatz(self, metadata: AnsatzMetadata) -> bool:
        """Check if this gradient strategy works with the given ansatz.

        ParameterShift: True only if metadata.gradient_compatibility
            includes "parameter_shift" or "both"
        FiniteDifference: always True (universal fallback)

        Returns:
            True if compatible, False otherwise.
        """
