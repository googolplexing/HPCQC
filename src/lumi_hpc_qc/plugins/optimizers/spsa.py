# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""SPSA optimizer — stochastic, 2 evals per step regardless of param count."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from lumi_hpc_qc.plugins.optimizers.base import OptimizerStrategy
from lumi_hpc_qc.types import ExperimentConfig, IterationRecord, OptimizeResult


class SpsaOptimizer(OptimizerStrategy):
    """Simultaneous Perturbation Stochastic Approximation.

    Only 2 function evaluations per step (regardless of parameter count).
    Ideal for noisy or shot-based evaluation where gradients are unreliable.
    """

    name = "spsa"
    requires_gradient = False

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        config: ExperimentConfig | None = None,
        callback: Callable[[IterationRecord], None] | None = None,
    ) -> OptimizeResult:
        p = config.optimizer_params if config else {}
        maxiter = p.get("maxiter", 200)
        a = p.get("a", 0.1)
        c = p.get("c", 0.1)
        A = p.get("A", maxiter * 0.1)
        alpha = p.get("alpha", 0.602)
        gamma = p.get("gamma", 0.101)

        params = x0.copy()
        best_energy = float('inf')
        best_params = params.copy()
        nfev = 0

        for k in range(maxiter):
            t0 = time.time()
            ak = a / (k + 1 + A) ** alpha
            ck = c / (k + 1) ** gamma

            # Random perturbation direction (Bernoulli ±1)
            delta = 2 * np.random.randint(0, 2, size=len(params)) - 1

            # Two-point gradient estimate
            e_plus = cost_fn(params + ck * delta)
            e_minus = cost_fn(params - ck * delta)
            nfev += 2

            ghat = (e_plus - e_minus) / (2 * ck * delta)
            params = params - ak * ghat

            energy = (e_plus + e_minus) / 2.0
            elapsed = time.time() - t0

            if energy < best_energy:
                best_energy = energy
                best_params = params.copy()

            if callback:
                callback(IterationRecord(
                    iteration=k + 1, energy=energy,
                    parameters=params.copy(),
                    gradient_norm=float(np.linalg.norm(ghat)),
                    elapsed_s=elapsed,
                ))

        return OptimizeResult(
            x=best_params, fun=best_energy,
            nfev=nfev, nit=maxiter,
            success=True, message="SPSA completed",
        )
