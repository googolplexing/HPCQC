# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Demultiplexer — extract per-placement results from composite circuits.

Given the raw measurement counts from a device-width composite circuit
and the placements that were packed into it, extracts per-placement
count distributions and computes per-placement expectation values.

Generalizes Phase C's ResultDemultiplexer.

RED-SPEC-002 §3 — Multi-Round Same-Circuit Packing
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lumi_hpc_qc.sweep.placement_solver import Placement


def demultiplex_counts(
    raw_counts: dict[str, int],
    placements: list[Placement],
    device_qubits: int,
) -> list[dict[str, int]]:
    """Extract per-placement count distributions from composite results.

    Args:
        raw_counts: Full device bitstring → count dict from composite execution.
        placements: The placements used in the composite circuit.
        device_qubits: Total device qubit count (e.g., 53).

    Returns:
        List of per-placement count dicts. Each maps N-qubit bitstrings
        to counts, where N = len(placement.physical_indices).
    """
    num_logical = len(placements[0].physical_indices) if placements else 0
    per_placement: list[dict[str, int]] = [{} for _ in placements]

    for bitstring, count in raw_counts.items():
        bits = bitstring.replace(" ", "").zfill(device_qubits)

        for p_idx, placement in enumerate(placements):
            phys = placement.physical_indices
            # Extract bits for this placement's physical qubits
            sub_bits = ""
            for logical_q in range(num_logical):
                phys_q = phys[logical_q]
                # Qiskit convention: bitstring position i = qubit[N-1-i]
                bit_pos = device_qubits - 1 - phys_q
                sub_bits += bits[bit_pos]

            if sub_bits in per_placement[p_idx]:
                per_placement[p_idx][sub_bits] += count
            else:
                per_placement[p_idx][sub_bits] = count

    return per_placement


def demultiplex_density_matrix(
    dm: Any,
    placements: list[Placement],
    device_qubits: int,
) -> list[Any]:
    """Extract per-placement reduced density matrices from composite DM.

    For exact evaluation (shots=0), the composite circuit's density matrix
    is a 2^D × 2^D matrix (D = device qubits). Each placement's reduced
    state is obtained by tracing out all qubits not in that placement.

    Args:
        dm: Full device density matrix (numpy array or DensityMatrix).
        placements: The placements in the composite circuit.
        device_qubits: Total device qubit count.

    Returns:
        List of reduced density matrices, one per placement.
    """
    dm_array = np.array(dm)

    reduced = []
    for placement in placements:
        phys = placement.physical_indices
        # Qubits to trace out = all device qubits NOT in this placement
        keep = set(phys)
        trace_out = [q for q in range(device_qubits) if q not in keep]

        # Partial trace: reshape and trace over non-placement qubits
        rdm = _partial_trace(dm_array, trace_out, device_qubits)
        reduced.append(rdm)

    return reduced


def compute_placement_energies(
    raw_counts: dict[str, int],
    placements: list[Placement],
    observable: Any,
    device_qubits: int,
) -> list[float]:
    """Full pipeline: demux → energy per placement from shot-based counts.

    Args:
        raw_counts: Composite circuit measurement counts.
        placements: Placements packed into the composite.
        observable: SparsePauliOp for expectation computation.
        device_qubits: Total device qubit count.

    Returns:
        List of energy values, one per placement.
    """
    per_counts = demultiplex_counts(raw_counts, placements, device_qubits)
    num_logical = len(placements[0].physical_indices) if placements else 0

    energies = []
    for counts in per_counts:
        energy = _energy_from_counts(counts, observable, num_logical)
        energies.append(energy)

    return energies


def compute_placement_energies_exact(
    dm: Any,
    placements: list[Placement],
    observable: Any,
    device_qubits: int,
) -> list[float]:
    """Compute per-placement energies from composite density matrix.

    Uses partial trace to get each placement's reduced DM, then
    computes Tr(H @ ρ) for each.

    Args:
        dm: Composite density matrix.
        placements: Placements in the composite.
        observable: SparsePauliOp.
        device_qubits: Total device qubits.

    Returns:
        List of energy values.
    """
    reduced_dms = demultiplex_density_matrix(dm, placements, device_qubits)
    h_matrix = observable.to_matrix()

    energies = []
    for rdm in reduced_dms:
        energy = float(np.real(np.trace(h_matrix @ rdm)))
        energies.append(energy)

    return energies


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _partial_trace(dm: np.ndarray, trace_out: list[int], n_qubits: int) -> np.ndarray:
    """Compute the partial trace of a density matrix.

    Traces out the specified qubits, returning the reduced density
    matrix for the remaining qubits.

    Args:
        dm: Density matrix as 2^n × 2^n numpy array.
        trace_out: List of qubit indices to trace out.
        n_qubits: Total number of qubits.

    Returns:
        Reduced density matrix for the kept qubits.
    """
    if not trace_out:
        return dm

    n_keep = n_qubits - len(trace_out)
    keep = sorted(set(range(n_qubits)) - set(trace_out))

    # Reshape to tensor with one axis per qubit (bra and ket)
    shape = [2] * (2 * n_qubits)
    rho = dm.reshape(shape)

    # Trace out qubits one at a time (from highest index to avoid shifts)
    for q in sorted(trace_out, reverse=True):
        # Contract bra and ket axes for qubit q
        # Bra axis = q, Ket axis = q + n_remaining
        n_remaining = rho.ndim // 2
        rho = np.trace(rho, axis1=q, axis2=q + n_remaining)

    # Reshape back to matrix
    dim = 2 ** n_keep
    return rho.reshape(dim, dim)


def _energy_from_counts(
    counts: dict[str, int],
    observable: Any,
    num_qubits: int,
) -> float:
    """Estimate ⟨H⟩ from Z-basis measurement counts — Z/I terms ONLY.

    DEPRECATED: This function only handles Z and I Pauli terms correctly.
    For QPU demultiplexed results with X/Y Hamiltonians, basis rotation
    must be applied at circuit construction time (in circuit_composer)
    before measurement. Use pauli_measurement module for correct handling.

    Raises ValueError if any X or Y terms are present to prevent silent
    wrong results.
    """
    # Guard: reject observables with non-Z/I terms
    for pauli_label in observable.paulis.to_labels():
        non_iz = set(pauli_label) - {"I", "Z"}
        if non_iz:
            raise ValueError(
                f"_energy_from_counts cannot estimate Pauli terms containing "
                f"{non_iz} from Z-basis measurements alone. For QPU results, "
                f"basis rotation must be applied at circuit construction time. "
                f"Use pauli_measurement.build_measurement_circuits() for "
                f"correct handling of X/Y terms."
            )

    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    energy = 0.0
    for pauli_label, coeff in zip(observable.paulis.to_labels(),
                                   observable.coeffs):
        parity_sum = 0.0
        for bitstring, count in counts.items():
            bits = bitstring[::-1]  # reverse for qiskit convention
            parity = 0
            for i, p in enumerate(pauli_label[::-1]):
                if p in ("Z",) and i < len(bits):
                    if bits[i] == "1":
                        parity ^= 1
            sign = 1 - 2 * parity
            parity_sum += sign * count
        energy += float(np.real(coeff)) * (parity_sum / total_shots)

    return energy
