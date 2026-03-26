# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Parameter-shift rule gradient — exact for standard Pauli rotation gates."""

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

    def compute(self, eval_fn: Callable, params: np.ndarray, backend=None) -> np.ndarray:
        grad = np.zeros(len(params))
        for k in range(len(params)):
            pp = params.copy()
            pp[k] += self._SHIFT
            pm = params.copy()
            pm[k] -= self._SHIFT
            grad[k] = (eval_fn(pp) - eval_fn(pm)) / 2.0
        return grad

    def validate_ansatz(self, metadata: AnsatzMetadata) -> bool:
        return metadata.gradient_compatibility in ("parameter_shift", "both")
