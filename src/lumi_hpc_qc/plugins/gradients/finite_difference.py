# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Finite-difference gradient — universal fallback for non-standard gates."""

from __future__ import annotations

from typing import Callable

import numpy as np

from lumi_hpc_qc.plugins.gradients.base import GradientStrategy
from lumi_hpc_qc.types import AnsatzMetadata


class FiniteDifferenceGradient(GradientStrategy):
    """Central finite-difference gradient.

    Uses central differences: (E(θ+eps) - E(θ-eps)) / (2*eps)
    More accurate than forward differences and handles saddle points better.

    Uses eps=0.1 because UCCSD excitation operators have small coefficients
    (e.g., 0.125*t). At θ≈0, a perturbation of eps=0.01 produces rotation
    changes of ~0.00125 rad — below the energy sensitivity threshold.
    eps=0.1 gives effective rotations of ~0.0125 rad with clean gradients.
    """

    name = "finite_difference"
    _EPS = 0.1

    @property
    def circuits_per_gradient(self) -> str:
        return "2n"

    def compute(self, eval_fn: Callable, params: np.ndarray, backend=None) -> np.ndarray:
        grad = np.zeros(len(params))
        for k in range(len(params)):
            pp = params.copy()
            pp[k] += self._EPS
            pm = params.copy()
            pm[k] -= self._EPS
            grad[k] = (eval_fn(pp) - eval_fn(pm)) / (2.0 * self._EPS)
        return grad

    def validate_ansatz(self, metadata: AnsatzMetadata) -> bool:
        return True  # universal fallback — works for any ansatz
