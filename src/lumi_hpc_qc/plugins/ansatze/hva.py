# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Hamiltonian Variational Ansatz (HVA) — for Fermi-Hubbard and Heisenberg."""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from lumi_hpc_qc.plugins.ansatze.base import AnsatzBuilder
from lumi_hpc_qc.types import AnsatzMetadata, ExperimentConfig


class HvaAnsatz(AnsatzBuilder):
    name = "hva"

    def build(self, num_qubits: int, config: ExperimentConfig) -> tuple[QuantumCircuit, AnsatzMetadata]:
        p = config.ansatz_params
        reps = p.get("reps", 3)
        model_type = config.model  # "fermi_hubbard" or "heisenberg"
        mp = config.model_params
        rows = mp.get("lattice_rows", 2)
        cols = mp.get("lattice_cols", 3)
        num_sites = rows * cols

        # Validate: circuit must have enough qubits for the lattice
        # Fermi-Hubbard uses 2× sites (up + down spin), Heisenberg uses 1× sites
        required_qubits = num_sites * 2 if model_type == "fermi_hubbard" else num_sites
        if num_qubits < required_qubits:
            raise ValueError(
                f"HVA ansatz requires {required_qubits} qubits for "
                f"{rows}×{cols} lattice ({model_type}), but num_qubits={num_qubits}. "
                f"Set model_params.lattice_rows/lattice_cols to match your qubit count."
            )

        # Build lattice edges
        edges = []
        for r in range(rows):
            for c in range(cols):
                s = r * cols + c
                if c + 1 < cols:
                    edges.append((s, s + 1))
                if r + 1 < rows:
                    edges.append((s, s + cols))

        # Count parameters per layer
        if model_type == "fermi_hubbard":
            params_per_layer = len(edges) * 2 + num_sites  # hopping(2 spins) + onsite
        else:
            params_per_layer = len(edges)  # one per edge for XXZ

        total_params = params_per_layer * reps + num_qubits  # + number-preserving RY
        params = ParameterVector('θ', total_params)
        qc = QuantumCircuit(num_qubits)

        # Initial state
        if model_type == "fermi_hubbard":
            for i in range(0, num_sites, 2):
                qc.x(i)
                qc.x(num_sites + i)
        else:
            for i in range(0, num_qubits, 2):
                qc.x(i)

        param_names = []
        idx = 0

        for rep in range(reps):
            if model_type == "fermi_hubbard":
                for (i, j) in edges:
                    qc.rxx(params[idx], i, j)
                    qc.ryy(params[idx], i, j)
                    param_names.append(f"L{rep}_hop_up({i},{j})")
                    idx += 1
                    qc.rxx(params[idx], i + num_sites, j + num_sites)
                    qc.ryy(params[idx], i + num_sites, j + num_sites)
                    param_names.append(f"L{rep}_hop_dn({i+num_sites},{j+num_sites})")
                    idx += 1
                for site in range(num_sites):
                    qc.rzz(params[idx], site, site + num_sites)
                    param_names.append(f"L{rep}_onsite_{site}")
                    idx += 1
            else:
                for (i, j) in edges:
                    qc.rxx(params[idx], i, j)
                    qc.ryy(params[idx], i, j)
                    qc.rzz(params[idx], i, j)
                    param_names.append(f"L{rep}_XXZ({i},{j})")
                    idx += 1

        # Number-preserving RY layer
        for q in range(num_qubits):
            qc.ry(params[idx], q)
            param_names.append(f"np_ry_{q}")
            idx += 1

        meta = AnsatzMetadata(
            num_parameters=total_params,
            parameter_names=param_names,
            gradient_compatibility="parameter_shift",
            preferred_initializer="adiabatic" if model_type in ("fermi_hubbard", "heisenberg") else "random",
            requires_decomposition=False,
            circuit_depth=qc.depth(),
            gate_counts=dict(qc.count_ops()),
        )
        return qc, meta
