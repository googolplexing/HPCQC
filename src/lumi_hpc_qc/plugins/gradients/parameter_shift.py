# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Parameter-shift rule gradient — exact for standard Pauli rotation gates.

Phase 3: Added batched mode. For n parameters, builds 2n shifted param
arrays. The workflow submits all 2n circuits to Aer in one sim.run() call,
distributing across available GPUs. For 64 params on 8 GPUs:
  Sequential: 128 × 0.8s = 102s per gradient
  Batched:    128 circuits / 8 GPUs ≈ 13s per gradient (~8× speedup)
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from lumi_hpc_qc.plugins.gradients.base import GradientStrategy
from lumi_hpc_qc.types import AnsatzMetadata


class ParameterShiftGradient(GradientStrategy):
    name = "parameter_shift"
    _SHIFT = np.pi / 2

    @property
    def circuits_per_gradient(self) -> str:
        return "2n"

    @property
    def supports_batching(self) -> bool:
        return True

    def compute(self, eval_fn: Callable, params: np.ndarray, backend=None) -> np.ndarray:
        """Sequential fallback — one circuit at a time."""
        grad = np.zeros(len(params))
        for k in range(len(params)):
            pp = params.copy()
            pp[k] += self._SHIFT
            pm = params.copy()
            pm[k] -= self._SHIFT
            grad[k] = (eval_fn(pp) - eval_fn(pm)) / 2.0
        return grad

    def build_shifted_params(self, params: np.ndarray) -> list[np.ndarray]:
        """Build 2n shifted parameter arrays for batch evaluation.

        Layout: [θ₁+, θ₁-, θ₂+, θ₂-, ..., θₙ+, θₙ-]
        where θₖ+ = params with params[k] += π/2
              θₖ- = params with params[k] -= π/2
        """
        shifted = []
        for k in range(len(params)):
            pp = params.copy()
            pp[k] += self._SHIFT
            shifted.append(pp)

            pm = params.copy()
            pm[k] -= self._SHIFT
            shifted.append(pm)
        return shifted

    def assemble_gradient(self, params: np.ndarray, energies: list[float]) -> np.ndarray:
        """Assemble gradient from 2n energies.

        energies[2k]   = E(θₖ + π/2)
        energies[2k+1] = E(θₖ - π/2)
        ∂E/∂θₖ = (E+ - E-) / 2
        """
        n = len(params)
        grad = np.zeros(n)
        for k in range(n):
            grad[k] = (energies[2 * k] - energies[2 * k + 1]) / 2.0
        return grad

    def validate_ansatz(self, metadata: AnsatzMetadata) -> bool:
        return metadata.gradient_compatibility in ("parameter_shift", "both")
