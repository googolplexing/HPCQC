# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""IQM QPU backend — VTT Q50 quantum processor via FiQCI middleware.

Submits circuits to the IQM quantum processor through CSC's HPC-Quantum
middleware. The Q50 is accessible from any SLURM partition with adequate
resources — not limited to q_fiqci.

References:
  - FiQCI blog: https://fiqci.fi/publications/2025-09-12-Simulating-Electrons
  - CSC docs: https://docs.csc.fi/computing/quantum-computing/running-quantum-jobs/
  - CSC Quantum repo: https://github.com/CSCfi/Quantum/tree/main/Variational-Algorithms-on-Q50

Phase 2 stub: implements the Backend interface so it can be selected
via config. Actual QPU submission requires iqm-client and FiQCI
credentials, which are only available on LUMI compute nodes.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lumi_hpc_qc.backends.base import Backend
from lumi_hpc_qc.types import (
    BackendCapabilities,
    CircuitJob,
    CircuitResult,
    ExperimentConfig,
)


class IqmQpuBackend(Backend):
    """IQM Q50 quantum processor backend via FiQCI."""

    name = "iqm_qpu"

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config
        self._sim = None  # not used; kept for interface compat
        self._use_blocking = False
        self._blocking_qubits = 0
        self._shots = 1024

        if config:
            bp = config.backend_params
            self._shots = bp.get("shots", 1024)
            self._url = bp.get("iqm_url", "")
            self._calibration_set_id = bp.get("calibration_set_id", None)

    def _ensure_sim(self) -> None:
        """Initialize IQM client connection.

        Requires iqm-client package and FiQCI environment variables.
        """
        if self._sim is not None:
            return

        try:
            from iqm.qiskit_iqm import IQMProvider
            provider = IQMProvider(self._url)
            self._sim = provider.get_backend()
            print(f"  Connected to IQM QPU: {self._sim.name}")
        except ImportError:
            raise ImportError(
                "IQM QPU backend requires 'iqm-client' and 'qiskit-iqm' packages.\n"
                "Install: pip install iqm-client qiskit-iqm\n"
                "These are available in the FiQCI container on LUMI."
            )

    def run_circuits(self, jobs: list[CircuitJob]) -> list[CircuitResult]:
        """Submit circuits to Q50 and collect measurement results.

        Note: QPU returns measurement counts, not expectation values.
        Energy must be computed from counts via:
          E = Σ_i coeff_i × <Z_i> where <Z_i> = (n_0 - n_1) / n_total
        """
        import time
        self._ensure_sim()
        results = []

        for job in jobs:
            t0 = time.time()
            energies = []

            for i, circuit in enumerate(job.circuits):
                if job.parameters and i < len(job.parameters):
                    bound = circuit.assign_parameters(job.parameters[i])
                else:
                    bound = circuit

                # QPU requires measurement gates
                if not any(inst.operation.name == 'measure' for inst in bound.data):
                    bound.measure_all()

                shots = job.shots if job.shots > 0 else self._shots
                qpu_result = self._sim.run(bound, shots=shots).result()
                counts = qpu_result.get_counts()

                # Compute expectation value from counts if observable provided
                if job.observable is not None:
                    energy = self._expectation_from_counts(
                        counts, job.observable, shots
                    )
                    energies.append(energy)

            elapsed = time.time() - t0
            results.append(CircuitResult(
                job_id=job.job_id,
                energies=energies if energies else None,
                counts=counts if not energies else None,
                execution_time_s=elapsed,
                backend_name=self.name,
            ))

        return results

    def compile_circuit(self, circuit):
        """Transpile circuit for IQM native gate set."""
        self._ensure_sim()
        from qiskit import transpile
        return transpile(circuit, backend=self._sim, optimization_level=2)

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=50,
            supports_statevector=False,
            supports_density_matrix=False,
            supports_mps=False,
            supports_shots=True,
            requires_gpu=False,
            requires_cpu_only=False,
            slurm_partition="q_fiqci",
        )

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        errors = []
        nq = config.num_qubits
        if nq and nq > 50:
            errors.append(f"IQM Q50 supports max 50 qubits, got {nq}")
        if not config.backend_params.get("iqm_url"):
            errors.append("IQM QPU requires 'iqm_url' in backend_params")
        return errors

    @staticmethod
    def _expectation_from_counts(counts: dict, observable, total_shots: int) -> float:
        """Compute ⟨H⟩ from measurement counts.

        For each Pauli term in the observable:
          ⟨P⟩ = Σ_bitstring (-1)^parity × count / total_shots
        where parity = number of 1s at positions where P has Z.
        """
        from qiskit.quantum_info import SparsePauliOp
        if not isinstance(observable, SparsePauliOp):
            observable = SparsePauliOp.from_operator(observable)

        energy = 0.0
        for pauli_label, coeff in zip(observable.to_list(), observable.coeffs):
            label = pauli_label[0] if isinstance(pauli_label, tuple) else str(pauli_label)

            if all(c in ('I', 'i') for c in label):
                energy += float(np.real(coeff))
                continue

            z_positions = [i for i, c in enumerate(reversed(label)) if c in ('Z', 'z')]
            if not z_positions:
                energy += float(np.real(coeff))
                continue

            expectation = 0.0
            for bitstring, count in counts.items():
                bits = bitstring.replace(' ', '')
                parity = sum(int(bits[-(pos + 1)]) for pos in z_positions if pos < len(bits))
                expectation += ((-1) ** parity) * count

            energy += float(np.real(coeff)) * expectation / total_shots

        return energy
