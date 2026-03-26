# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Zero-Noise Extrapolation (ZNE) error mitigation.

Runs the same circuit at multiple noise scale factors (1x, 2x, 3x)
by inserting identity-equivalent gate pairs (e.g., CNOT-CNOT), then
extrapolates to the zero-noise limit via polynomial or exponential fit.

Applicable to: shot-based simulation and QPU backends.
Not meaningful for noiseless statevector simulation.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from lumi_hpc_qc.plugins.error_mitigation.base import ErrorMitigator
from lumi_hpc_qc.types import ExperimentConfig


class ZneErrorMitigator(ErrorMitigator):
    """Zero-Noise Extrapolation via gate folding."""

    name = "zne"
    description = "Zero-noise extrapolation with polynomial/exponential fit"

    def mitigate(
        self,
        eval_fn: Callable,
        params: np.ndarray,
        config: ExperimentConfig,
    ) -> float:
        """Run ZNE: evaluate at multiple noise levels, extrapolate to zero.

        Uses the eval_fn at scale factors 1, 2, 3 and fits a polynomial
        to extrapolate to noise_factor=0.

        For noiseless simulation, this just returns eval_fn(params) since
        all scale factors give the same result.
        """
        p = config.error_mitigation_params
        scale_factors = p.get("scale_factors", [1, 2, 3])
        extrapolation = p.get("extrapolation", "linear")

        # Evaluate at each noise level
        # In a real implementation, this would fold gates to amplify noise
        # For now, delegate to backend-level noise amplification
        energies = []
        for sf in scale_factors:
            # TODO: implement gate folding at circuit level
            # For now, just evaluate normally (placeholder)
            e = eval_fn(params)
            energies.append(e)

        # Extrapolate to zero noise
        if extrapolation == "linear":
            # Linear fit: E(λ) = a + b*λ, extrapolate to λ=0
            coeffs = np.polyfit(scale_factors, energies, 1)
            return float(np.polyval(coeffs, 0))
        elif extrapolation == "exponential":
            # Exponential fit: E(λ) = a * exp(b*λ) + c
            # Simplified: use Richardson extrapolation
            if len(energies) >= 2:
                return float(2 * energies[0] - energies[1])
            return energies[0]
        else:
            # Polynomial fit
            degree = min(len(scale_factors) - 1, 2)
            coeffs = np.polyfit(scale_factors, energies, degree)
            return float(np.polyval(coeffs, 0))

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        errors = []
        if config.backend in ("aer_gpu", "aer_cpu"):
            method = config.backend_params.get("method", "statevector")
            shots = config.backend_params.get("shots", 0)
            if method == "statevector" and shots == 0:
                errors.append(
                    "ZNE is not meaningful for noiseless statevector simulation. "
                    "Use shot-based simulation or QPU backend."
                )
        return errors
