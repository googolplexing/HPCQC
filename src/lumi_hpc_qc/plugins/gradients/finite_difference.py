# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Finite-difference gradient — universal fallback for non-standard gates.

Phase 3: Added batched mode using central differences.
For n parameters, builds 2n shifted param arrays (same as parameter-shift).
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from lumi_hpc_qc.plugins.gradients.base import GradientStrategy
from lumi_hpc_qc.types import AnsatzMetadata


class FiniteDifferenceGradient(GradientStrategy):
    """Central finite-difference gradient.

    Uses eps=0.1 because UCCSD excitation operators have small coefficients
    (e.g., 0.125*t). At θ≈0, a perturbation of eps=0.01 produces rotation
    changes of ~0.00125 rad — below the energy sensitivity threshold.
    """

    name = "finite_difference"
    _EPS = 0.1

    @property
    def circuits_per_gradient(self) -> str:
        return "2n"

    @property
    def supports_batching(self) -> bool:
        return True

    def compute(self, eval_fn: Callable, params: np.ndarray, backend=None) -> np.ndarray:
        """Sequential fallback — central differences, one at a time."""
        grad = np.zeros(len(params))
        for k in range(len(params)):
            pp = params.copy()
            pp[k] += self._EPS
            pm = params.copy()
            pm[k] -= self._EPS
            grad[k] = (eval_fn(pp) - eval_fn(pm)) / (2.0 * self._EPS)
        return grad

    def build_shifted_params(self, params: np.ndarray) -> list[np.ndarray]:
        """Build 2n shifted parameter arrays for batch evaluation.

        Layout: [θ₁+ε, θ₁-ε, θ₂+ε, θ₂-ε, ..., θₙ+ε, θₙ-ε]
        """
        shifted = []
        for k in range(len(params)):
            pp = params.copy()
            pp[k] += self._EPS
            shifted.append(pp)

            pm = params.copy()
            pm[k] -= self._EPS
            shifted.append(pm)
        return shifted

    def assemble_gradient(self, params: np.ndarray, energies: list[float]) -> np.ndarray:
        """Assemble gradient from 2n energies using central differences.

        energies[2k]   = E(θₖ + ε)
        energies[2k+1] = E(θₖ - ε)
        ∂E/∂θₖ ≈ (E+ - E-) / 2ε
        """
        n = len(params)
        grad = np.zeros(n)
        for k in range(n):
            grad[k] = (energies[2 * k] - energies[2 * k + 1]) / (2.0 * self._EPS)
        return grad

    def validate_ansatz(self, metadata: AnsatzMetadata) -> bool:
        return True
