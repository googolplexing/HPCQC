# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Molecular electronic structure Hamiltonian builder."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from lumi_hpc_qc.plugins.hamiltonians.base import HamiltonianBuilder
from lumi_hpc_qc.types import ExperimentConfig, HamiltonianMetadata

# Hardcoded H2 STO-3G coefficients (works without PySCF)
_H2_STO3G_PAULIS = [
    ("IIII", -0.81054), ("IIIZ", 0.17218), ("IIZI", -0.22575),
    ("IZII", 0.17218), ("ZIII", -0.22575), ("IIZZ", 0.12091),
    ("IZIZ", 0.16892), ("IZZI", 0.04523), ("ZIIZ", 0.04523),
    ("ZIZI", 0.16892), ("ZZII", 0.17464), ("XXYY", -0.04523),
    ("XYYX", 0.04523), ("YXXY", 0.04523), ("YYXX", -0.04523),
]


class MolecularHamiltonian(HamiltonianBuilder):
    name = "molecular"
    description = "Molecular electronic structure via PySCF + Jordan-Wigner"

    def build(self, config: ExperimentConfig) -> tuple[SparsePauliOp, HamiltonianMetadata]:
        p = config.model_params
        molecule = p.get("molecule", "H2")
        basis = p.get("basis", "sto-3g")
        distance = p.get("distance", 0.735)
        mapper_type = p.get("mapper_type", "jordan_wigner")

        try:
            ham, nq, desc = self._build_pyscf(molecule, basis, distance, mapper_type, p)
        except (ImportError, Exception):
            if molecule.upper() == "H2":
                ham = SparsePauliOp.from_list(_H2_STO3G_PAULIS).simplify()
                nq, desc = 4, "H2 STO-3G (hardcoded coefficients)"
            else:
                raise

        meta = HamiltonianMetadata(
            num_qubits=nq, num_pauli_terms=len(ham), qubit_mapping=mapper_type,
            description=f"Molecular VQE: {molecule}, {basis} basis, {nq} qubits",
            physical_params={"molecule": molecule, "basis": basis, "distance": distance},
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

    @staticmethod
    def _build_pyscf(molecule, basis, distance, mapper_type, p):
        from qiskit_nature.second_q.drivers import PySCFDriver
        from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper
        from qiskit_nature.units import DistanceUnit

        geometries = {
            "H2": [("H", [0, 0, 0]), ("H", [0, 0, distance])],
            "LIH": [("Li", [0, 0, 0]), ("H", [0, 0, 1.6])],
            "H2O": [("O", [0, 0, 0]), ("H", [0.757, 0.586, 0]), ("H", [-0.757, 0.586, 0])],
        }
        mol_upper = molecule.upper()
        h_match = re.match(r'^H(\d+)$', mol_upper)
        if h_match and int(h_match.group(1)) > 2:
            n = int(h_match.group(1))
            d = distance if distance != 0.735 else 0.74
            geometries[mol_upper] = [("H", [0, 0, i * d]) for i in range(n)]

        if mol_upper in geometries:
            geom_str = "; ".join(f"{a} {x} {y} {z}" for a, (x, y, z) in geometries[mol_upper])
        else:
            geom_str = molecule

        driver = PySCFDriver(atom=geom_str, basis=basis,
                             charge=p.get("charge", 0), spin=p.get("spin", 0),
                             unit=DistanceUnit.ANGSTROM)
        problem = driver.run()
        ham_2q = problem.hamiltonian.second_q_op()
        mapper = ParityMapper(num_particles=problem.num_particles) if mapper_type == "parity" else JordanWignerMapper()
        qh = mapper.map(ham_2q)
        if not isinstance(qh, SparsePauliOp):
            qh = SparsePauliOp.from_operator(qh)
        qh = qh.simplify()
        return qh, qh.num_qubits, f"{molecule} {basis}"
