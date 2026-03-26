# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""L-BFGS-B optimizer — gradient-based, good for noiseless simulation."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
from scipy.optimize import minimize

from lumi_hpc_qc.plugins.optimizers.base import OptimizerStrategy
from lumi_hpc_qc.types import ExperimentConfig, IterationRecord, OptimizeResult


class LbfgsbOptimizer(OptimizerStrategy):
    name = "l_bfgs_b"
    requires_gradient = True

    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        config: ExperimentConfig | None = None,
        callback: Callable[[IterationRecord], None] | None = None,
    ) -> OptimizeResult:
        if grad_fn is None:
            raise ValueError("L-BFGS-B requires a gradient function")

        p = config.optimizer_params if config else {}
        maxiter = p.get("maxiter", 1000)
        gtol = p.get("gtol", 1e-6)

        iteration_count = [0]

        def cost_and_grad(params):
            t0 = time.time()
            energy = cost_fn(params)
            grad = grad_fn(params)
            elapsed = time.time() - t0
            iteration_count[0] += 1

            if callback:
                callback(IterationRecord(
                    iteration=iteration_count[0],
                    energy=energy,
                    parameters=params.copy(),
                    gradient_norm=float(np.linalg.norm(grad)),
                    elapsed_s=elapsed,
                ))
            return energy, grad

        result = minimize(
            cost_and_grad, x0, method='L-BFGS-B', jac=True,
            options={'maxiter': maxiter, 'gtol': gtol},
        )

        return OptimizeResult(
            x=result.x, fun=result.fun,
            nfev=result.nfev, nit=result.nit,
            success=result.success, message=str(result.message),
        )
