# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Abstract base class for optimization strategies.

Wraps scipy.optimize, qiskit_algorithms, or custom optimizers behind
a uniform interface. The workflow calls minimize() and receives an
OptimizeResult — it never imports scipy directly.

To add a new optimizer:
  1. Create a new .py file in plugins/optimizers/
  2. Subclass OptimizerStrategy
  3. Implement minimize()
  4. Set requires_gradient = True if the optimizer needs gradients
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from lumi_hpc_qc.types import ExperimentConfig, IterationRecord, OptimizeResult


class OptimizerStrategy(ABC):
    """Abstract optimization strategy."""

    name: str = ""
    requires_gradient: bool = False  # True for L-BFGS-B/BFGS, False for COBYLA/SPSA

    @abstractmethod
    def minimize(
        self,
        cost_fn: Callable[[np.ndarray], float],
        x0: np.ndarray,
        grad_fn: Callable[[np.ndarray], np.ndarray] | None = None,
        config: ExperimentConfig | None = None,
        callback: Callable[[IterationRecord], None] | None = None,
    ) -> OptimizeResult:
        """Run optimization loop.

        Args:
            cost_fn: Evaluates energy E(θ). Called with parameter array,
                returns scalar energy.
            x0: Initial parameter vector.
            grad_fn: Computes gradient ∇E(θ). Required if requires_gradient
                is True. Called with parameter array, returns gradient array.
            config: Experiment config for optimizer hyperparameters
                (maxiter, gtol, learning_rate, etc.)
            callback: Called after every iteration with an IterationRecord.
                The workflow uses this for checkpointing and logging.
                May be None (no callback).

        Returns:
            OptimizeResult with optimal parameters, energy, and diagnostics.

        Contract:
            - If requires_gradient and grad_fn is None, raise ValueError
            - callback is called with is_best=True when a new best is found
            - cost_fn and grad_fn may be slow (GPU simulation) — optimizer
              should minimize the number of calls
        """
