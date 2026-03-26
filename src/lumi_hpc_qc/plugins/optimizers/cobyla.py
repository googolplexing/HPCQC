# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""COBYLA optimizer — derivative-free, for noisy or shot-based evaluation."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from lumi_hpc_qc.plugins.optimizers.base import OptimizerStrategy
from lumi_hpc_qc.types import ExperimentConfig, IterationRecord, OptimizeResult


class CobylaOptimizer(OptimizerStrategy):
    name = "cobyla"
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
        rhobeg = p.get("rhobeg", 0.3)

        iteration_count = [0]

        def wrapped_cost(params):
            t0 = time.time()
            energy = cost_fn(params)
            elapsed = time.time() - t0
            iteration_count[0] += 1
            if callback:
                callback(IterationRecord(
                    iteration=iteration_count[0],
                    energy=energy,
                    parameters=params.copy(),
                    elapsed_s=elapsed,
                ))
            return energy

        result = minimize(
            wrapped_cost, x0, method='COBYLA',
            options={'maxiter': maxiter, 'rhobeg': rhobeg},
        )

        return OptimizeResult(
            x=result.x, fun=result.fun,
            nfev=result.nfev, nit=getattr(result, 'nit', iteration_count[0]),
            success=result.success, message=str(result.message),
        )
