# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Multiplexed QPU execution — pack multiple circuit copies onto Q50.

BLUE-RESP-001 §11: Q50 executes all 53 qubits every shot regardless of
how many a circuit uses. A 4-qubit VQE submission leaves 49 qubits idle.
Circuit multiplexing packs up to 12 copies of a 4-qubit circuit onto
non-overlapping qubit subsets, generating 12× data per submission.

Components:
  1. MultiplexedCircuitBuilder: N sub-circuits + N placements → one 53q circuit
  2. ResultDemultiplexer: 53q bitstring counts → N per-placement count dicts
  3. Per-placement energy computation via pauli_measurement

Usage:
    from lumi_hpc_qc.plugins.placement.multiplexer import MultiplexedCircuitBuilder

    builder = MultiplexedCircuitBuilder(device_qubits=53)
    mux_circuit = builder.build(sub_circuit, placements)
    # ... execute on QPU ...
    per_placement_counts = builder.demultiplex(raw_counts, placements)
"""

from __future__ import annotations

from typing import Any

import numpy as np


class MultiplexedCircuitBuilder:
    """Build and demultiplex circuits for multiplexed QPU execution."""

    def __init__(self, device_qubits: int = 53) -> None:
        self._device_qubits = device_qubits

    def build(
        self,
        sub_circuit,
        placements: list[dict],
        params: np.ndarray | None = None,
    ):
        """Build a multiplexed circuit from sub-circuit + placements.

        Args:
            sub_circuit: The parameterized ansatz circuit (N qubits).
            placements: List of placement dicts from PlacementSolver,
                each with 'physical_indices' mapping.
            params: Optional parameter values to bind before multiplexing.

        Returns:
            A single QuantumCircuit on device_qubits qubits with all
            placements composed in parallel.
        """
        from qiskit import QuantumCircuit

        mux = QuantumCircuit(self._device_qubits, self._device_qubits)
        num_logical = sub_circuit.num_qubits

        for p in placements:
            phys = p["physical_indices"]
            if len(phys) != num_logical:
                raise ValueError(
                    f"Placement has {len(phys)} qubits but circuit has {num_logical}"
                )

            # Bind parameters if provided
            if params is not None and sub_circuit.num_parameters > 0:
                param_dict = dict(zip(sub_circuit.parameters, params))
                bound = sub_circuit.assign_parameters(param_dict)
            else:
                bound = sub_circuit

            # Map logical qubits to physical qubits
            qubit_map = {i: phys[i] for i in range(num_logical)}

            # Append gates with remapped qubits
            for instruction in bound.data:
                op = instruction.operation
                qubits = [mux.qubits[qubit_map[bound.qubits.index(q)]]
                          for q in instruction.qubits]
                clbits = []  # measurements handled separately
                mux.append(op, qubits, clbits)

        # Add measurements for all used qubits
        used_qubits = set()
        for p in placements:
            used_qubits.update(p["physical_indices"])
        for q in sorted(used_qubits):
            mux.measure(q, q)

        return mux

    def demultiplex(
        self,
        raw_counts: dict[str, int],
        placements: list[dict],
        num_logical_qubits: int,
    ) -> list[dict[str, int]]:
        """Extract per-placement counts from multiplexed measurement results.

        Args:
            raw_counts: Full device bitstring → count dict.
            placements: Same placements used in build().
            num_logical_qubits: Number of qubits per sub-circuit.

        Returns:
            List of per-placement count dicts, one per placement.
            Each dict maps N-qubit bitstrings to counts.
        """
        per_placement = [{} for _ in placements]

        for bitstring, count in raw_counts.items():
            bits = bitstring.replace(" ", "")
            # Qiskit bitstring: MSB...LSB, bit[i] corresponds to qubit[N-1-i]
            device_bits = bits.zfill(self._device_qubits)

            for p_idx, placement in enumerate(placements):
                phys = placement["physical_indices"]
                # Extract bits for this placement's physical qubits
                sub_bits = ""
                for logical_q in range(num_logical_qubits):
                    phys_q = phys[logical_q]
                    # Qiskit convention: bit at position i = qubit[N-1-i]
                    bit_pos = self._device_qubits - 1 - phys_q
                    sub_bits += device_bits[bit_pos]

                if sub_bits in per_placement[p_idx]:
                    per_placement[p_idx][sub_bits] += count
                else:
                    per_placement[p_idx][sub_bits] = count

        return per_placement

    def compute_per_placement_energies(
        self,
        raw_counts: dict[str, int],
        placements: list[dict],
        observable,
        num_logical_qubits: int,
        total_shots: int,
        readout_fidelities: list[list[float]] | None = None,
    ) -> list[dict]:
        """Full pipeline: demultiplex → optional readout correction → energy.

        Args:
            raw_counts: Full device measurement counts.
            placements: Placements from solver.
            observable: SparsePauliOp Hamiltonian.
            num_logical_qubits: Circuit qubit count.
            total_shots: Shots per execution.
            readout_fidelities: Optional per-placement readout fidelities
                for readout error mitigation.

        Returns:
            List of dicts, one per placement:
                energy: float
                counts: dict
                placement_id: int
                physical_qubits: list[str]
        """
        from lumi_hpc_qc.backends.pauli_measurement import (
            build_measurement_circuits,
            expectation_from_grouped_counts,
        )

        per_counts = self.demultiplex(raw_counts, placements, num_logical_qubits)
        results = []

        for p_idx, (counts, placement) in enumerate(zip(per_counts, placements)):
            # Optional readout correction
            if readout_fidelities and p_idx < len(readout_fidelities):
                from lumi_hpc_qc.plugins.error_mitigation.readout import ReadoutMitigator
                mitigator = ReadoutMitigator()
                counts = mitigator.correct_counts(
                    counts, readout_fidelities[p_idx], total_shots
                )

            # Compute energy from counts using direct expectation
            # (measurement basis rotation was applied before multiplexing)
            from lumi_hpc_qc.backends.pauli_measurement import (
                expectation_from_counts_direct,
            )
            energy = expectation_from_counts_direct(counts, observable, total_shots)

            results.append({
                "placement_id": placement.get("placement_id", p_idx),
                "energy": float(energy),
                "counts": counts,
                "physical_qubits": placement.get("qubit_mapping", {}),
                "physical_indices": placement.get("physical_indices", []),
                "avg_readout_fidelity": placement.get("avg_readout_fidelity", 0),
                "avg_cz_fidelity": placement.get("avg_cz_fidelity", 0),
            })

        return results

    @staticmethod
    def build_placement_metadata(
        placements: list[dict],
        calibration_path: str,
    ) -> list[dict]:
        """Build per-placement metadata including calibration data.

        RED-SPEC-001-v1.1 §B.3: Each placement's trajectory must include
        physical qubit assignment and per-qubit calibration data.
        """
        import json
        with open(calibration_path) as f:
            cal = json.load(f)

        qubits_data = cal.get("qubits", {})
        qubit_names = list(qubits_data.keys())

        metadata = []
        for p in placements:
            phys_indices = p.get("physical_indices", [])
            qubit_mapping = p.get("qubit_mapping", {})

            # Per-qubit calibration for this placement
            per_qubit_cal = {}
            for logical_q, phys_name in qubit_mapping.items():
                if phys_name in qubits_data:
                    per_qubit_cal[f"logical_{logical_q}"] = {
                        "physical_qubit": phys_name,
                        **qubits_data[phys_name],
                    }

            metadata.append({
                "placement_id": p.get("placement_id", 0),
                "qubit_mapping": qubit_mapping,
                "physical_indices": phys_indices,
                "per_qubit_calibration": per_qubit_cal,
                "avg_readout_fidelity": p.get("avg_readout_fidelity", 0),
                "avg_cz_fidelity": p.get("avg_cz_fidelity", 0),
                "internal_edges": p.get("internal_edges", 0),
                "score": p.get("score", 0),
            })

        return metadata
