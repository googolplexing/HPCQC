# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Fermi-Hubbard model Hamiltonian builder."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from lumi_hpc_qc.plugins.hamiltonians.base import HamiltonianBuilder
from lumi_hpc_qc.types import ExperimentConfig, HamiltonianMetadata


class FermiHubbardHamiltonian(HamiltonianBuilder):
    name = "fermi_hubbard"
    description = "2D Fermi-Hubbard model with Jordan-Wigner encoding"

    def build(self, config: ExperimentConfig) -> tuple[SparsePauliOp, HamiltonianMetadata]:
        p = config.model_params
        rows = p.get("lattice_rows", 2)
        cols = p.get("lattice_cols", 3)
        t = p.get("hopping_t", 1.0)
        u = p.get("interaction_u", 2.0)
        bc = p.get("boundary_condition", "open")

        ham, raw_meta = self._build_fh(rows, cols, t, u, bc)
        meta = HamiltonianMetadata(
            num_qubits=raw_meta["num_qubits"],
            pauli_term_count=raw_meta["num_pauli_terms"],
            qubit_mapping="jordan_wigner",
            description=raw_meta["description"],
            physical_params={"lattice_rows": rows, "lattice_cols": cols,
                             "hopping_t": t, "interaction_u": u, "u_over_t": u / t if t else 0},
        )
        return ham, meta

    def exact_ground_energy(self, hamiltonian: Any) -> float | None:
        if hamiltonian.num_qubits > 24:
            return None
        mat = hamiltonian.to_matrix()
        eigenvalues = np.linalg.eigvalsh(mat)
        return float(np.real(eigenvalues[0]))

    def adiabatic_parameter_name(self) -> str | None:
        return "interaction_u"

    def build_at_parameter(self, value: float, config: ExperimentConfig) -> SparsePauliOp:
        p = config.model_params
        ham, _ = self._build_fh(
            p.get("lattice_rows", 2), p.get("lattice_cols", 3),
            p.get("hopping_t", 1.0), value, p.get("boundary_condition", "open"),
        )
        return ham

    @staticmethod
    def _build_fh(rows, cols, t, u, bc):
        from qiskit_nature.second_q.hamiltonians import FermiHubbardModel
        from qiskit_nature.second_q.hamiltonians.lattices import (
            BoundaryCondition, SquareLattice, LineLattice,
        )
        from qiskit_nature.second_q.mappers import JordanWignerMapper

        boundary = (BoundaryCondition.PERIODIC if bc == "periodic"
                     else BoundaryCondition.OPEN)
        n_sites = rows * cols
        if rows == 1:
            lattice = LineLattice(num_nodes=cols, boundary_condition=boundary)
        elif cols == 1:
            lattice = LineLattice(num_nodes=rows, boundary_condition=boundary)
        else:
            lattice = SquareLattice(rows=rows, cols=cols, boundary_condition=boundary)

        fhm = FermiHubbardModel(
            lattice.uniform_parameters(uniform_interaction=t, uniform_onsite_potential=0.0),
            onsite_interaction=u,
        )
        ham_2q = fhm.second_q_op()
        mapper = JordanWignerMapper()
        qh = mapper.map(ham_2q)
        if not isinstance(qh, SparsePauliOp):
            qh = SparsePauliOp.from_operator(qh)
        qh = qh.simplify()
        nq = qh.num_qubits
        meta = {
            "num_qubits": nq, "num_pauli_terms": len(qh),
            "description": (f"2D Fermi-Hubbard model on {rows}x{cols} lattice, "
                            f"U/t={u/t:.2f}, Jordan-Wigner encoding, {nq} qubits"),
        }
        return qh, meta
