# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Pauli basis measurement utilities — correct expectation values from counts.

F1 FIX: The original _expectation_from_counts() only handled Z-basis parity,
silently dropping X and Y Pauli term contributions. This module implements
proper basis rotation before measurement:
  - Z positions: measure directly (computational basis)
  - X positions: apply H gate, then measure in Z basis
  - Y positions: apply Sdg then H, then measure in Z basis

After rotation, all measurements are in the Z basis, and Z-parity
correctly computes the expectation value for any Pauli string.

Usage:
    from lumi_hpc_qc.backends.pauli_measurement import (
        build_measurement_circuits,
        expectation_from_grouped_counts,
    )
"""

from __future__ import annotations

from typing import Any

import numpy as np


def group_commuting_paulis(observable) -> list[dict]:
    """Group Pauli terms into qubit-wise commuting (QWC) sets.

    Two Pauli strings qubit-wise commute if, at every qubit position,
    they either have the same Pauli operator or at least one is I.

    Returns a list of groups, each containing:
        - labels: list of Pauli strings
        - coeffs: list of complex coefficients
        - basis: dict mapping qubit_index -> 'X'|'Y'|'Z' (the measurement basis)
    """
    from qiskit.quantum_info import SparsePauliOp
    if not isinstance(observable, SparsePauliOp):
        observable = SparsePauliOp.from_operator(observable)

    labels = observable.paulis.to_labels()
    coeffs = list(observable.coeffs)

    groups = []

    for label, coeff in zip(labels, coeffs):
        # Skip pure identity terms — they contribute coeff directly
        if all(c == 'I' for c in label):
            groups.append({
                "labels": [label],
                "coeffs": [coeff],
                "basis": {},
                "is_identity": True,
            })
            continue

        placed = False
        for group in groups:
            if group.get("is_identity"):
                continue
            # Check qubit-wise commutativity with existing basis
            compatible = True
            for pos, pauli_char in enumerate(reversed(label)):
                if pauli_char == 'I':
                    continue
                existing = group["basis"].get(pos)
                if existing is not None and existing != pauli_char:
                    compatible = False
                    break
            if compatible:
                group["labels"].append(label)
                group["coeffs"].append(coeff)
                # Update basis map
                for pos, pauli_char in enumerate(reversed(label)):
                    if pauli_char != 'I':
                        group["basis"][pos] = pauli_char
                placed = True
                break

        if not placed:
            basis = {}
            for pos, pauli_char in enumerate(reversed(label)):
                if pauli_char != 'I':
                    basis[pos] = pauli_char
            groups.append({
                "labels": [label],
                "coeffs": [coeff],
                "basis": basis,
                "is_identity": False,
            })

    return groups


def build_measurement_circuits(
    bound_circuit,
    observable,
    shots: int,
) -> tuple[list, list[dict], float]:
    """Build basis-rotated measurement circuits for an observable.

    For each group of qubit-wise commuting Pauli terms:
    1. Copy the state-preparation circuit
    2. Apply basis rotation gates (H for X, Sdg+H for Y)
    3. Add Z-basis measurement on all qubits

    Args:
        bound_circuit: The parameter-bound ansatz circuit (no measurements).
        observable: SparsePauliOp Hamiltonian.
        shots: Number of measurement shots per circuit.

    Returns:
        (circuits, groups, identity_contribution)
        - circuits: list of QuantumCircuit with measurements
        - groups: list of group dicts (for expectation computation)
        - identity_contribution: sum of coefficients of pure-I terms
    """
    from qiskit import QuantumCircuit

    groups = group_commuting_paulis(observable)

    circuits = []
    measurement_groups = []
    identity_contribution = 0.0

    for group in groups:
        if group.get("is_identity"):
            identity_contribution += float(np.real(group["coeffs"][0]))
            continue

        # Build rotated measurement circuit
        nq = bound_circuit.num_qubits
        meas_circuit = bound_circuit.copy()

        # Apply basis rotation for each qubit that needs non-Z measurement
        for qubit_pos, pauli_char in group["basis"].items():
            if qubit_pos >= nq:
                continue
            if pauli_char == 'X':
                meas_circuit.h(qubit_pos)
            elif pauli_char == 'Y':
                meas_circuit.sdg(qubit_pos)
                meas_circuit.h(qubit_pos)
            # Z: no rotation needed

        meas_circuit.measure_all()
        circuits.append(meas_circuit)
        measurement_groups.append(group)

    return circuits, measurement_groups, identity_contribution


def expectation_from_grouped_counts(
    counts_list: list[dict],
    measurement_groups: list[dict],
    identity_contribution: float,
    total_shots: int,
) -> float:
    """Compute ⟨H⟩ from basis-rotated measurement counts.

    After basis rotation, every qubit is measured in the Z basis.
    The expectation of each Pauli term is the parity of the
    relevant qubit positions in each bitstring.

    Args:
        counts_list: One counts dict per measurement circuit.
        measurement_groups: Groups from build_measurement_circuits.
        identity_contribution: Sum of pure-identity coefficients.
        total_shots: Shots per circuit.

    Returns:
        float: The expectation value ⟨H⟩.
    """
    energy = identity_contribution

    for counts, group in zip(counts_list, measurement_groups):
        actual_shots = sum(counts.values())
        if actual_shots == 0:
            continue

        for label, coeff in zip(group["labels"], group["coeffs"]):
            # Find positions with non-I operators
            # After basis rotation, all contribute Z-parity
            active_positions = [
                i for i, c in enumerate(reversed(label)) if c != 'I'
            ]

            if not active_positions:
                energy += float(np.real(coeff))
                continue

            expectation = 0.0
            for bitstring, count in counts.items():
                bits = bitstring.replace(' ', '')
                # Compute parity of active positions
                parity = 0
                for pos in active_positions:
                    if pos < len(bits):
                        parity += int(bits[-(pos + 1)])
                expectation += ((-1) ** parity) * count

            energy += float(np.real(coeff)) * expectation / actual_shots

    return energy


def expectation_from_counts_direct(
    counts: dict,
    observable,
    total_shots: int,
) -> float:
    """Compute ⟨H⟩ from a SINGLE set of Z-basis counts.

    WARNING: This only works correctly if the circuit was measured in
    the Z basis AND the observable contains only Z and I terms.
    For mixed X/Y/Z observables, use build_measurement_circuits +
    expectation_from_grouped_counts instead.

    This function exists for backward compatibility with code that
    already handles its own basis rotation externally.
    """
    from qiskit.quantum_info import SparsePauliOp
    if not isinstance(observable, SparsePauliOp):
        observable = SparsePauliOp.from_operator(observable)

    energy = 0.0
    for pauli, coeff in zip(observable.paulis.to_labels(), observable.coeffs):
        if all(c == 'I' for c in pauli):
            energy += float(np.real(coeff))
            continue

        # All non-I positions contribute to parity (after rotation, all are Z)
        active_positions = [
            i for i, c in enumerate(reversed(pauli)) if c != 'I'
        ]

        # Check for non-Z terms — warn if present
        has_non_z = any(c not in ('I', 'Z') for c in pauli)
        if has_non_z:
            import warnings
            warnings.warn(
                f"expectation_from_counts_direct called with non-Z Pauli "
                f"term '{pauli}'. Results will be wrong unless basis rotation "
                f"was applied externally. Use build_measurement_circuits() "
                f"for correct handling of X/Y terms.",
                stacklevel=2,
            )

        expectation = 0.0
        for bitstring, count in counts.items():
            bits = bitstring.replace(' ', '')
            parity = sum(
                int(bits[-(pos + 1)]) for pos in active_positions
                if pos < len(bits)
            )
            expectation += ((-1) ** parity) * count

        energy += float(np.real(coeff)) * expectation / total_shots

    return energy
