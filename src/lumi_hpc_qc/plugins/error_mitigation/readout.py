# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Readout error mitigation via calibration matrix inversion.

Corrects measurement errors using the tensor product structure of
independent single-qubit readout errors. For N qubits, the full
calibration matrix is 2^N × 2^N, but independent readout errors
decompose as M = M_1 ⊗ M_2 ⊗ ... ⊗ M_N, so the inversion is
N independent 2×2 inversions applied per-qubit — O(N × 2^N) instead
of O(2^2N).

Each single-qubit calibration matrix uses the symmetric model (C5):
    M_k = [[1-p_k, p_k], [p_k, 1-p_k]]
    where p_k = (1 - readout_fidelity_k) / 2

The correction maps raw count probabilities to mitigated probabilities:
    p_mitigated = M^{-1} @ p_raw

Negative probabilities from the inversion are clipped to zero and
the distribution is renormalized.

Usage:
    from lumi_hpc_qc.plugins.error_mitigation.readout import ReadoutMitigator

    mitigator = ReadoutMitigator()
    corrected_counts = mitigator.correct_counts(
        raw_counts={"00": 900, "01": 50, "10": 40, "11": 10},
        readout_fidelities=[0.97, 0.95],
        total_shots=1000,
    )
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from lumi_hpc_qc.plugins.error_mitigation.base import ErrorMitigator
from lumi_hpc_qc.types import CircuitResult, ExperimentConfig


class ReadoutMitigator(ErrorMitigator):
    """Readout error mitigation via tensor product calibration matrix inversion."""

    name = "readout"

    def correct_counts(
        self,
        raw_counts: dict[str, int],
        readout_fidelities: list[float],
        total_shots: int,
    ) -> dict[str, int]:
        """Apply readout error correction to measurement counts.

        Args:
            raw_counts: Bitstring → count dict from measurement.
            readout_fidelities: Per-qubit readout fidelities [q0, q1, ...].
            total_shots: Total number of shots.

        Returns:
            Corrected counts dict (same format as input).
        """
        num_qubits = len(readout_fidelities)
        num_states = 2 ** num_qubits

        # Convert counts to probability vector
        probs = np.zeros(num_states)
        for bitstring, count in raw_counts.items():
            bits = bitstring.replace(" ", "")
            idx = int(bits, 2)
            probs[idx] = count / total_shots

        # Apply per-qubit M^{-1} iteratively
        # For each qubit k, reshape probs and apply 2×2 correction
        corrected = probs.copy()
        for k in range(num_qubits):
            fid = readout_fidelities[k]
            p_error = (1 - fid) / 2

            # 2×2 inverse calibration matrix for qubit k
            # M = [[1-p, p], [p, 1-p]]
            # M^{-1} = (1/(1-2p)) * [[1-p, -p], [-p, 1-p]]
            det = 1 - 2 * p_error
            if abs(det) < 1e-12:
                continue  # skip if fidelity ≈ 0.5 (uninformative)

            m_inv = np.array([
                [1 - p_error, -p_error],
                [-p_error, 1 - p_error],
            ]) / det

            # Reshape to apply M^{-1}_k along qubit k's axis
            # Qubit k splits the 2^N states into pairs differing only at bit k
            shape = [2] * num_qubits
            corrected = corrected.reshape(shape)

            # Apply along axis k (from MSB: axis 0 = qubit 0 = leftmost bit)
            corrected = np.tensordot(m_inv, corrected, axes=([1], [k]))
            # tensordot moves the contracted axis to position 0 — move it back
            corrected = np.moveaxis(corrected, 0, k)

            corrected = corrected.flatten()

        # Clip negative probabilities and renormalize
        corrected = np.maximum(corrected, 0)
        total = corrected.sum()
        if total > 0:
            corrected /= total

        # Convert back to counts
        result = {}
        for idx in range(num_states):
            count = int(round(corrected[idx] * total_shots))
            if count > 0:
                bitstring = format(idx, f'0{num_qubits}b')
                result[bitstring] = count

        return result

    def load_fidelities(
        self,
        calibration_path: str,
        num_qubits: int,
    ) -> list[float]:
        """Load per-qubit readout fidelities from calibration file.

        Uses the same connectivity-aware qubit selection as the noise model.
        """
        from lumi_hpc_qc.backends.noise_model import _load_calibration, _select_qubits

        cal = _load_calibration(calibration_path)
        selected = _select_qubits(cal, num_qubits)
        return [qdata["readout_fidelity"] for _, qdata in selected]

    def pre_process(self, circuits, config):
        """Readout mitigation doesn't need additional circuits."""
        return circuits

    def post_process(self, results, config):
        """Apply readout correction to circuit results.

        This operates on the counts inside CircuitResult before
        energy computation.
        """
        # Readout correction is applied at the counts level,
        # integrated into the energy evaluation pipeline.
        # See correct_counts() for the actual correction.
        return results
