# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Hardware-efficient EfficientSU2 ansatz."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from lumi_hpc_qc.plugins.ansatze.base import AnsatzBuilder
from lumi_hpc_qc.types import AnsatzMetadata, ExperimentConfig


class Su2Ansatz(AnsatzBuilder):
    name = "su2"

    def build(self, num_qubits: int, config: ExperimentConfig) -> tuple[QuantumCircuit, AnsatzMetadata]:
        from qiskit.circuit.library import EfficientSU2

        p = config.ansatz_params
        reps = p.get("reps", 3)
        entanglement = p.get("entanglement", "linear")

        su2 = EfficientSU2(num_qubits=num_qubits, reps=reps, entanglement=entanglement)

        # Multi-round decomposition to primitive gates (Aer requirement)
        prev_depth = su2.depth()
        for _ in range(10):
            su2 = su2.decompose()
            if su2.depth() == prev_depth:
                break
            prev_depth = su2.depth()

        total_params = su2.num_parameters
        param_names = [f"su2_{i}" for i in range(total_params)]

        meta = AnsatzMetadata(
            num_parameters=total_params,
            parameter_names=param_names,
            gradient_compatibility="parameter_shift",
            preferred_initializer="random",
            requires_decomposition=False,  # already decomposed above
            circuit_depth=su2.depth(),
            gate_counts=dict(su2.count_ops()),
        )
        return su2, meta
