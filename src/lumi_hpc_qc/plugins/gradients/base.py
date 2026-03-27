# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Abstract base class for gradient computation strategies.

Phase 3 additions:
  - supports_batching property: declares if this strategy can batch circuits
  - compute_batched(): builds all shifted param arrays, returns them for
    batch evaluation. The workflow submits all circuits in one sim.run() call,
    then the strategy assembles the gradient from the results.

The workflow auto-selects batched mode when:
  1. The gradient strategy declares supports_batching = True
  2. The backend supports batch circuit submission (Aer GPU/CPU)

Sequential fallback (compute()) is always preserved for backends that
don't support batch submission (e.g., QPU with shot-based evaluation).
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
        """Human-readable description of circuit cost."""

    @property
    def supports_batching(self) -> bool:
        """Whether this strategy can produce batched parameter arrays.

        If True, the workflow can call build_shifted_params() to get all
        parameter arrays at once, evaluate them in a single batch, then
        call assemble_gradient() to compute the gradient from results.

        Default: False (sequential compute() only).
        """
        return False

    @abstractmethod
    def compute(
        self,
        eval_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
        backend: Backend | None = None,
    ) -> np.ndarray:
        """Compute gradient vector sequentially (one eval at a time).

        This is the fallback path — always works, any backend.
        """

    def build_shifted_params(self, params: np.ndarray) -> list[np.ndarray]:
        """Build all shifted parameter arrays for batched evaluation.

        Returns a list of parameter arrays. The workflow evaluates all of
        them in a single batch call, then passes the energies to
        assemble_gradient().

        Only called when supports_batching is True.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batched gradient. "
            "Set supports_batching = True and implement build_shifted_params() "
            "and assemble_gradient()."
        )

    def assemble_gradient(
        self, params: np.ndarray, energies: list[float]
    ) -> np.ndarray:
        """Assemble gradient from batch-evaluated energies.

        Args:
            params: Original parameter vector (for reference).
            energies: Energies corresponding to build_shifted_params() output,
                      in the same order.

        Returns:
            Gradient array of shape (n_params,).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support batched gradient."
        )

    @abstractmethod
    def validate_ansatz(self, metadata: AnsatzMetadata) -> bool:
        """Check if this gradient strategy works with the given ansatz."""
