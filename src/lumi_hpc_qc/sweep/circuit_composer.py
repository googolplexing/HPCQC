# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Circuit composer — pack multiple placements into one device-width circuit.

Given a round of non-overlapping placements and a circuit template,
builds a single QuantumCircuit spanning all device qubits with each
placement's circuit mapped to its physical qubits. Barriers separate
placements for clarity but don't affect execution.

Generalizes Phase C's MultiplexedCircuitBuilder to work with Phase E
Placement objects from the VF2 solver.

RED-SPEC-002 §3 — Multi-Round Same-Circuit Packing
"""

from __future__ import annotations

from typing import Any

from qiskit import QuantumCircuit

from lumi_hpc_qc.sweep.placement_solver import Placement


def compose_round(
    circuit: QuantumCircuit,
    placements: list[Placement],
    device_qubits: int,
    *,
    params: Any | None = None,
    add_measurements: bool = True,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build a device-width circuit with all placements composed in parallel.

    Args:
        circuit: The template circuit (N logical qubits).
        placements: Non-overlapping placements for one round.
        device_qubits: Total qubits on the device (e.g., 53 for Q50).
        params: Optional parameter values to bind before composing.
        add_measurements: If True, add measurements on all used qubits.
        add_barriers: If True, add barriers between placements.

    Returns:
        A single QuantumCircuit on device_qubits qubits.

    Raises:
        ValueError: If placements overlap in qubits.
    """
    # Verify non-overlapping
    all_used = set()
    for p in placements:
        qubit_set = set(p.physical_indices)
        overlap = all_used & qubit_set
        if overlap:
            raise ValueError(
                f"Placement {p.placement_id} overlaps with previous "
                f"placements on qubits {overlap}"
            )
        all_used.update(qubit_set)

    # Build composite circuit
    n_clbits = device_qubits if add_measurements else 0
    composite = QuantumCircuit(device_qubits, n_clbits)

    num_logical = circuit.num_qubits

    for p_idx, placement in enumerate(placements):
        phys = placement.physical_indices
        if len(phys) != num_logical:
            raise ValueError(
                f"Placement has {len(phys)} qubits but circuit has "
                f"{num_logical}"
            )

        # Bind parameters if provided
        if params is not None and circuit.num_parameters > 0:
            param_dict = dict(zip(circuit.parameters, params))
            bound = circuit.assign_parameters(param_dict)
        else:
            bound = circuit

        # Map logical → physical and append gates
        qubit_map = {i: phys[i] for i in range(num_logical)}
        for instruction in bound.data:
            op = instruction.operation
            # Skip measurement instructions — we add our own
            if op.name == "measure":
                continue
            # Skip save_* instructions (density matrix etc.)
            if op.name.startswith("save_"):
                continue

            mapped_qubits = [
                composite.qubits[qubit_map[bound.find_bit(q).index]]
                for q in instruction.qubits
            ]
            composite.append(op, mapped_qubits, [])

        if add_barriers and p_idx < len(placements) - 1:
            composite.barrier()

    # Add measurements on all used qubits
    if add_measurements:
        for q in sorted(all_used):
            composite.measure(q, q)

    return composite


def compose_round_for_density_matrix(
    circuit: QuantumCircuit,
    placements: list[Placement],
    device_qubits: int,
    *,
    params: Any | None = None,
) -> QuantumCircuit:
    """Build composite circuit with save_density_matrix for exact evaluation.

    Same as compose_round but adds save_density_matrix instead of
    measurements. Used for exact (shots=0) Aer evaluation.
    """
    composite = compose_round(
        circuit, placements, device_qubits,
        params=params,
        add_measurements=False,
        add_barriers=True,
    )
    composite.save_density_matrix()
    return composite
