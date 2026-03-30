# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""QAOA MaxCut Hamiltonian builder."""

from __future__ import annotations

from typing import Any

import numpy as np
from qiskit.quantum_info import SparsePauliOp

from lumi_hpc_qc.plugins.hamiltonians.base import HamiltonianBuilder
from lumi_hpc_qc.types import ExperimentConfig, HamiltonianMetadata


class QaoaMaxcutHamiltonian(HamiltonianBuilder):
    name = "qaoa_maxcut"
    description = "QAOA MaxCut cost Hamiltonian"

    def build(self, config: ExperimentConfig) -> tuple[SparsePauliOp, HamiltonianMetadata]:
        p = config.model_params
        n = p.get("num_nodes", 8)
        seed = p.get("seed", 42)
        degree = p.get("degree", 3)
        graph_type = p.get("graph_type", "random_regular")
        edge_list = p.get("edge_list", None)

        rng = np.random.RandomState(seed)
        if edge_list is None:
            edges = self._generate_graph(n, graph_type, degree, rng)
        else:
            edges = [tuple(e) for e in edge_list]

        pauli_list = []
        # MaxCut: C = Σ (1-ZiZj)/2 counts edges cut (positive, larger=better).
        # VQE MINIMIZES, so we need H = -C = Σ ZiZj/2 - |E|/2.
        # Ground state of H corresponds to the best cut.
        for (i, j) in edges:
            label = ['I'] * n
            label[n - 1 - i] = 'Z'
            label[n - 1 - j] = 'Z'
            pauli_list.append((''.join(label), 0.5))    # +ZiZj/2
        pauli_list.append(('I' * n, -len(edges) / 2.0)) # -|E|/2

        ham = SparsePauliOp.from_list(pauli_list).simplify()
        meta = HamiltonianMetadata(
            num_qubits=n, num_pauli_terms=len(ham), qubit_mapping="direct",
            description=f"QAOA MaxCut {graph_type} graph, {n} nodes, {len(edges)} edges",
            physical_params={"num_nodes": n, "num_edges": len(edges),
                             "edge_list": edges, "graph_type": graph_type},
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
    def _generate_graph(n, graph_type, degree, rng):
        if graph_type == "random_regular":
            edges = set()
            deg_count = [0] * n
            for _ in range(10000):
                i, j = rng.randint(0, n), rng.randint(0, n)
                if i != j and (i, j) not in edges and (j, i) not in edges:
                    if deg_count[i] < degree and deg_count[j] < degree:
                        edges.add((min(i, j), max(i, j)))
                        deg_count[i] += 1
                        deg_count[j] += 1
                if all(d >= degree for d in deg_count):
                    break
            return list(edges)
        elif graph_type == "complete":
            return [(i, j) for i in range(n) for j in range(i + 1, n)]
        else:
            rows = int(np.sqrt(n))
            while n % rows != 0 and rows > 1:
                rows -= 1
            cols = n // rows
            edges = []
            for r in range(rows):
                for c in range(cols):
                    node = r * cols + c
                    if c + 1 < cols: edges.append((node, node + 1))
                    if r + 1 < rows: edges.append((node, node + cols))
            return edges
