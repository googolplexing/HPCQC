# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Bring-Your-Own ansatz from QASM file."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from lumi_hpc_qc.plugins.ansatze.base import AnsatzBuilder
from lumi_hpc_qc.types import AnsatzMetadata, ExperimentConfig


class ByoAnsatz(AnsatzBuilder):
    name = "byo"

    def build(self, num_qubits: int, config: ExperimentConfig) -> tuple[QuantumCircuit, AnsatzMetadata]:
        qasm_file = config.ansatz_params.get("qasm_file")
        if not qasm_file:
            raise ValueError("BYO ansatz requires 'qasm_file' in ansatz_params")

        try:
            from qiskit.qasm3 import load as qasm3_load
            qc = qasm3_load(qasm_file)
        except Exception:
            qc = QuantumCircuit.from_qasm_file(qasm_file)

        n = qc.num_parameters
        meta = AnsatzMetadata(
            num_parameters=n,
            parameter_names=[f"byo_{i}" for i in range(n)],
            gradient_compatibility="parameter_shift",
            preferred_initializer="random",
            requires_decomposition=True,
            circuit_depth=qc.depth(),
            gate_counts=dict(qc.count_ops()),
        )
        return qc, meta
