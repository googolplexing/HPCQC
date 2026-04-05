# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Transverse-Field Ising Model (TFIM) Hamiltonian builder.

H = -J Σ_{<i,j>} Z_i Z_j  -  g Σ_i X_i

The primary benchmark model for the HPCQC project. Supports open and
periodic boundary conditions on 1D chains and 2D grids.

model_params:
    num_qubits: int — total qubit count (overrides lattice_rows × lattice_cols)
    lattice_rows: int — rows (default 1 for a chain)
    lattice_cols: int — columns (default num_qubits for a chain)
    j: float — Ising coupling strength (default 1.0)
    g: float — transverse field strength (default 1.0)
    boundary_condition: "open" | "periodic" (default "open")
"""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from lumi_hpc_qc.plugins.hamiltonians.base import HamiltonianBuilder
from lumi_hpc_qc.types import ExperimentConfig, HamiltonianMetadata


class TFIMHamiltonian(HamiltonianBuilder):
    name = "tfim"
    description = "Transverse-Field Ising Model — ZZ coupling + X field"

    def default_params(self, num_qubits: int) -> dict[str, Any]:
        return {"num_qubits": num_qubits, "j": 1.0, "g": 1.0,
                "boundary_condition": "open"}

    def build(self, config: ExperimentConfig) -> tuple[SparsePauliOp, HamiltonianMetadata]:
        p = config.model_params
        n = config.num_qubits
        rows = p.get("lattice_rows", 1)
        cols = p.get("lattice_cols", n)
        j = p.get("j", 1.0)
        g = p.get("g", 1.0)
        bc = p.get("boundary_condition", "open")

        # If num_qubits is set explicitly, override lattice dimensions
        if rows * cols != n:
            rows = 1
            cols = n

        ham = self._build_tfim(rows, cols, j, g, bc)
        nq = ham.num_qubits
        meta = HamiltonianMetadata(
            num_qubits=nq,
            num_pauli_terms=len(ham),
            qubit_mapping="direct",
            description=f"TFIM {rows}x{cols} ({nq}q), J={j}, g={g}, bc={bc}",
            physical_params={"j": j, "g": g, "boundary_condition": bc},
        )
        return ham, meta

    def exact_ground_energy(self, hamiltonian: Any) -> float | None:
        if hamiltonian.num_qubits > 24:
            return None
        return float(np.real(np.linalg.eigvalsh(hamiltonian.to_matrix())[0]))

    def adiabatic_parameter_name(self) -> str | None:
        return "g"

    def build_at_parameter(self, value: float, config: ExperimentConfig) -> SparsePauliOp:
        p = config.model_params
        n = config.num_qubits
        rows = p.get("lattice_rows", 1)
        cols = p.get("lattice_cols", n)
        if rows * cols != n:
            rows = 1
            cols = n
        return self._build_tfim(
            rows, cols,
            p.get("j", 1.0), value,
            p.get("boundary_condition", "open"),
        )

    @staticmethod
    def _build_tfim(rows: int, cols: int, j: float, g: float, bc: str) -> SparsePauliOp:
        """Construct the TFIM SparsePauliOp.

        Uses the same Pauli label convention as the Heisenberg plugin:
        label[n-1-i] is qubit i (big-endian, Qiskit standard).
        """
        n = rows * cols

        # Build edges from lattice
        edges: list[tuple[int, int]] = []
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                # Horizontal
                if c + 1 < cols:
                    edges.append((s, s + 1))
                elif bc == "periodic" and cols > 2:
                    edges.append((s, r * cols))
                # Vertical
                if r + 1 < rows:
                    edges.append((s, s + cols))
                elif bc == "periodic" and rows > 2:
                    edges.append((s, c))

        pauli_list: list[tuple[str, float]] = []

        # ZZ terms on edges
        for i, k in edges:
            label = ["I"] * n
            label[n - 1 - i] = "Z"
            label[n - 1 - k] = "Z"
            pauli_list.append(("".join(label), -j))

        # X terms on each qubit
        for i in range(n):
            label = ["I"] * n
            label[n - 1 - i] = "X"
            pauli_list.append(("".join(label), -g))

        return SparsePauliOp.from_list(pauli_list).simplify()
