# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Bring-Your-Own Hamiltonian from JSON file."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from lumi_hpc_qc.plugins.hamiltonians.base import HamiltonianBuilder
from lumi_hpc_qc.types import ExperimentConfig, HamiltonianMetadata


class ByoHamiltonian(HamiltonianBuilder):
    name = "byo"
    description = "User-defined Hamiltonian from JSON file or inline Pauli list"

    def build(self, config: ExperimentConfig) -> tuple[SparsePauliOp, HamiltonianMetadata]:
        p = config.model_params
        ham_file = p.get("hamiltonian_file")
        pauli_list = p.get("pauli_list")

        if pauli_list:
            terms = [(t[0], t[1]) for t in pauli_list]
        elif ham_file:
            with open(ham_file) as f:
                data = json.load(f)
            raw = data["pauli_terms"]
            terms = []
            for t in raw:
                if isinstance(t, dict):
                    terms.append((t["label"], t.get("coeff_real", 0) + 1j * t.get("coeff_imag", 0)))
                else:
                    terms.append((t[0], t[1]))
        else:
            raise ValueError("BYO model requires 'hamiltonian_file' or 'pauli_list' in model_params")

        ham = SparsePauliOp.from_list(terms).simplify()
        desc = f"User-defined, {ham.num_qubits} qubits, {len(ham)} terms"
        if ham_file:
            with open(ham_file) as f:
                data = json.load(f)
            desc = data.get("description", desc)

        meta = HamiltonianMetadata(
            num_qubits=ham.num_qubits, num_pauli_terms=len(ham),
            qubit_mapping="user_defined", description=desc,
            physical_params={"source": ham_file or "inline"},
        )
        return ham, meta

    def exact_ground_energy(self, hamiltonian: Any) -> float | None:
        if hamiltonian.num_qubits > 24:
            return None
        return float(np.real(np.linalg.eigvalsh(hamiltonian.to_matrix())[0]))

    def adiabatic_parameter_name(self) -> str | None:
        return None

    def build_at_parameter(self, value: float, config: ExperimentConfig) -> SparsePauliOp:
        ham, _ = self.build(config)
        return ham
