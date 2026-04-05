# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Heisenberg XXZ model Hamiltonian builder."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from lumi_hpc_qc.plugins.hamiltonians.base import HamiltonianBuilder
from lumi_hpc_qc.types import ExperimentConfig, HamiltonianMetadata


class HeisenbergHamiltonian(HamiltonianBuilder):
    name = "heisenberg"
    description = "Heisenberg XXZ model — direct spin-to-qubit mapping"

    def default_params(self, num_qubits: int) -> dict[str, Any]:
        return {"lattice_rows": 1, "lattice_cols": num_qubits,
                "jx": 1.0, "jy": 1.0, "jz": 1.0,
                "boundary_condition": "open"}

    def build(self, config: ExperimentConfig) -> tuple[SparsePauliOp, HamiltonianMetadata]:
        p = config.model_params
        ham = self._build_heis(
            p.get("lattice_rows", 3), p.get("lattice_cols", 4),
            p.get("jx", 1.0), p.get("jy", 1.0), p.get("jz", 1.0),
            p.get("h_field", 0.0), p.get("boundary_condition", "open"),
        )
        nq = ham.num_qubits
        meta = HamiltonianMetadata(
            num_qubits=nq, num_pauli_terms=len(ham), qubit_mapping="direct",
            description=f"Heisenberg on {p.get('lattice_rows',3)}x{p.get('lattice_cols',4)}, {nq} qubits",
            physical_params={k: p.get(k) for k in ["jx","jy","jz","h_field"]},
        )
        return ham, meta

    def exact_ground_energy(self, hamiltonian: Any) -> float | None:
        if hamiltonian.num_qubits > 24:
            return None
        return float(np.real(np.linalg.eigvalsh(hamiltonian.to_matrix())[0]))

    def adiabatic_parameter_name(self) -> str | None:
        return "jz"

    def build_at_parameter(self, value: float, config: ExperimentConfig) -> SparsePauliOp:
        p = config.model_params
        return self._build_heis(
            p.get("lattice_rows", 3), p.get("lattice_cols", 4),
            p.get("jx", 1.0), p.get("jy", 1.0), value,
            p.get("h_field", 0.0), p.get("boundary_condition", "open"),
        )

    @staticmethod
    def _build_heis(rows, cols, jx, jy, jz, h_field, bc):
        n = rows * cols
        edges = []
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                if c + 1 < cols:
                    edges.append((s, s + 1))
                elif bc == "periodic" and cols > 2:
                    edges.append((s, r * cols))
                if r + 1 < rows:
                    edges.append((s, s + cols))
                elif bc == "periodic" and rows > 2:
                    edges.append((s, c))
        pauli_list = []
        for (i, j) in edges:
            for pauli_char, coupling in [('X', jx), ('Y', jy), ('Z', jz)]:
                if coupling != 0:
                    label = ['I'] * n
                    label[n - 1 - i] = pauli_char
                    label[n - 1 - j] = pauli_char
                    pauli_list.append((''.join(label), coupling))
        if h_field != 0:
            for i in range(n):
                label = ['I'] * n
                label[n - 1 - i] = 'Z'
                pauli_list.append((''.join(label), h_field))
        return SparsePauliOp.from_list(pauli_list).simplify()
