# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Random parameter initialization — uniform [-π/2, π/2]."""

from __future__ import annotations

import numpy as np

from lumi_hpc_qc.plugins.initializers.base import InitializerStrategy
from lumi_hpc_qc.types import ExperimentConfig


class RandomInitializer(InitializerStrategy):
    name = "random"

    def initialize(self, num_params, hamiltonian_builder=None, ansatz=None,
                   backend=None, config=None):
        p = config.initializer_params if config else {}
        seed = p.get("seed", 42)
        low = p.get("low", -np.pi / 2)
        high = p.get("high", np.pi / 2)

        rng = np.random.RandomState(seed)
        return rng.uniform(low, high, num_params)
