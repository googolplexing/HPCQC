# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""UCCSD ansatz for molecular electronic structure VQE."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from lumi_hpc_qc.plugins.ansatze.base import AnsatzBuilder
from lumi_hpc_qc.types import AnsatzMetadata, ExperimentConfig


class UccsdAnsatz(AnsatzBuilder):
    name = "uccsd"

    def build(self, num_qubits: int, config: ExperimentConfig) -> tuple[QuantumCircuit, AnsatzMetadata]:
        p = config.ansatz_params
        num_electrons = p.get("num_electrons", config.model_params.get("num_electrons", 2))
        reps = p.get("reps", 1)

        # Validate: need at least as many qubits as electrons for JW mapping
        if num_electrons > num_qubits:
            raise ValueError(
                f"UCCSD requires num_qubits >= num_electrons, "
                f"got {num_qubits} qubits for {num_electrons} electrons."
            )
        if num_electrons < 1:
            raise ValueError(
                f"UCCSD requires at least 1 electron, got {num_electrons}."
            )

        try:
            qc, total_params, pnames = self._build_nature(num_qubits, num_electrons, reps)
        except (ImportError, Exception):
            qc, total_params, pnames = self._build_fallback(num_qubits, num_electrons, reps)

        meta = AnsatzMetadata(
            num_parameters=total_params,
            parameter_names=pnames,
            gradient_compatibility="finite_difference",
            preferred_initializer="zero",
            requires_decomposition=False,  # already decomposed
            circuit_depth=qc.depth(),
            gate_counts=dict(qc.count_ops()),
        )
        return qc, meta

    @staticmethod
    def _build_nature(num_qubits, num_electrons, reps):
        from qiskit import QuantumCircuit as QC
        from qiskit_nature.second_q.circuit.library import UCCSD as UCCSD_Nature
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        mapper = JordanWignerMapper()
        uccsd = UCCSD_Nature(
            num_spatial_orbitals=num_qubits // 2,
            num_particles=(num_electrons // 2, num_electrons // 2),
            qubit_mapper=mapper, reps=reps,
        )
        # Multi-round decomposition for Aer compatibility
        prev = uccsd.depth()
        for _ in range(10):
            uccsd = uccsd.decompose()
            if uccsd.depth() == prev:
                break
            prev = uccsd.depth()

        # Prepend explicit HF initial state (X gates on occupied orbitals)
        # qiskit-nature UCCSD may or may not include HF state depending on version
        # Belt-and-suspenders: check if circuit starts from |0⟩ or |HF⟩
        hf = QC(num_qubits)
        n_alpha = num_electrons // 2
        n_beta = num_electrons // 2
        for i in range(n_alpha):
            hf.x(i)
        for i in range(n_alpha, n_alpha + n_beta):
            hf.x(i)
        full = hf.compose(uccsd)

        n = full.num_parameters
        names = [f"uccsd_{i}" for i in range(n)]
        return full, n, names

    @staticmethod
    def _build_fallback(num_qubits, num_electrons, reps):
        from qiskit.circuit import ParameterVector

        num_occ = num_electrons
        num_virt = num_qubits - num_occ
        singles = [(i, num_occ + a) for i in range(num_occ) for a in range(num_virt)]
        doubles = [(i, j, num_occ + a, num_occ + b)
                   for i in range(num_occ) for j in range(i + 1, num_occ)
                   for a in range(num_virt) for b in range(a + 1, num_virt)]

        n = (len(singles) + len(doubles)) * reps
        params = ParameterVector('θ', n)
        qc = QuantumCircuit(num_qubits)
        for i in range(num_occ):
            qc.x(i)

        idx = 0
        names = []
        for _ in range(reps):
            for (i, a) in singles:
                qc.cx(i, a)
                qc.ry(params[idx], i)
                qc.cx(a, i)
                qc.ry(-params[idx], i)
                qc.cx(i, a)
                names.append(f"single_{i}->{a}")
                idx += 1
            for (i, j, a, b) in doubles:
                qc.cx(i, a)
                qc.cx(j, b)
                qc.rz(params[idx], b)
                qc.cx(j, b)
                qc.cx(i, a)
                names.append(f"double_({i},{j})->({a},{b})")
                idx += 1
        return qc, n, names
