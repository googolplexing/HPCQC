# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Zero parameter initialization — Hartree-Fock reference for UCCSD.

At θ=0, the UCCSD ansatz produces the HF ground state. The optimizer
then finds electron correlation corrections as small parameter perturbations.
"""

from __future__ import annotations

import numpy as np

from lumi_hpc_qc.plugins.initializers.base import InitializerStrategy
from lumi_hpc_qc.types import ExperimentConfig


class ZeroInitializer(InitializerStrategy):
    name = "zero"

    def initialize(self, num_params, hamiltonian_builder=None, ansatz=None,
                   backend=None, config=None):
        return np.zeros(num_params)
