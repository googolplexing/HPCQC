# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""QAOA ansatz for MaxCut."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from lumi_hpc_qc.plugins.ansatze.base import AnsatzBuilder
from lumi_hpc_qc.types import AnsatzMetadata, ExperimentConfig


class QaoaAnsatz(AnsatzBuilder):
    name = "qaoa"

    def build(self, num_qubits: int, config: ExperimentConfig) -> tuple[QuantumCircuit, AnsatzMetadata]:
        p = config.ansatz_params
        p_layers = p.get("p_layers", p.get("reps", 2))
        edge_list = config.model_params.get("edge_list", [])

        # Validate: all edge vertices must be within circuit bounds
        if edge_list:
            max_vertex = max(max(i, j) for i, j in edge_list)
            if max_vertex >= num_qubits:
                raise ValueError(
                    f"QAOA edge_list references qubit {max_vertex}, "
                    f"but circuit has only {num_qubits} qubits."
                )

        total_params = 2 * p_layers
        params = ParameterVector('θ', total_params)
        qc = QuantumCircuit(num_qubits)
        qc.h(range(num_qubits))

        param_names = []
        idx = 0
        for layer in range(p_layers):
            gamma = params[idx]
            beta = params[idx + 1]
            for (i, j) in edge_list:
                qc.rzz(gamma, i, j)
            param_names.append(f"L{layer}_gamma")
            idx += 1
            for q in range(num_qubits):
                qc.rx(2 * beta, q)
            param_names.append(f"L{layer}_beta")
            idx += 1

        meta = AnsatzMetadata(
            num_parameters=total_params,
            parameter_names=param_names,
            gradient_compatibility="parameter_shift",
            preferred_initializer="random",
            requires_decomposition=False,
            circuit_depth=qc.depth(),
            gate_counts=dict(qc.count_ops()),
        )
        return qc, meta
