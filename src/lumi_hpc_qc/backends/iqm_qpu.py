# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""IQM QPU backend — VTT Q50 quantum processor via FiQCI middleware.

Submits circuits to the IQM Q50 quantum processor through CSC's
HPC-Quantum middleware on LUMI. No API token required — access is
controlled through SLURM's q_fiqci partition and FiQCI modules.

Access pattern (from CSC docs):
    module use /appl/local/quantum/modulefiles
    module load fiqci-vtt-qiskit
    export DEVICES=("Q50")
    srun --partition q_fiqci -c 1 -n 1 bash -c "source $RUN_SETUP && python script.py"

The Q50_CORTEX_URL environment variable is set automatically by the
FiQCI module when running inside the q_fiqci partition. No manual
URL configuration needed.

References:
  - CSC docs: https://docs.csc.fi/computing/quantum-computing/running-quantum-jobs/
  - FiQCI blog: https://fiqci.fi/publications/2025-09-12-Simulating-Electrons
  - CSC Quantum repo: https://github.com/CSCfi/Quantum/tree/main/Variational-Algorithms-on-Q50
"""

from __future__ import annotations

import os
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
    """IQM Q50 quantum processor backend via FiQCI.

    When running inside the q_fiqci SLURM partition with the fiqci-vtt-qiskit
    module loaded, Q50_CORTEX_URL is set automatically. The backend connects
    to the QPU through this URL using IQMProvider from qiskit-iqm.

    Backend params (YAML):
        backend: iqm_qpu
        backend_params:
            shots: 4096            # measurement shots (default 1024)
            device: Q50            # Q50 or Q5 (default Q50)
            calibration_set_id: x  # optional, for reproducibility
    """

    name = "iqm_qpu"

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config
        self._sim = None  # IQM backend object (via IQMProvider)
        self._use_blocking = False
        self._blocking_qubits = 0
        self._shots = 1024
        self._device = "Q50"
        self._calibration_set_id = None

        if config:
            bp = config.backend_params
            self._shots = bp.get("shots", 1024)
            self._device = bp.get("device", "Q50")
            self._calibration_set_id = bp.get("calibration_set_id", None)

    def _ensure_sim(self) -> None:
        """Initialize IQM backend connection via FiQCI environment.

        The Q50_CORTEX_URL (or HELMI_CORTEX_URL for Q5) is set by the
        fiqci-vtt-qiskit module when running in the q_fiqci partition.
        No manual URL or token configuration needed.
        """
        if self._sim is not None:
            return

        # Determine URL from environment (set by FiQCI module)
        if self._device == "Q50":
            url_env = "Q50_CORTEX_URL"
        else:
            url_env = "HELMI_CORTEX_URL"

        url = os.getenv(url_env, "")

        if not url:
            # Not in q_fiqci partition or module not loaded
            print(f"  WARNING: {url_env} not set. To use {self._device}:")
            print(f"    module use /appl/local/quantum/modulefiles")
            print(f"    module load fiqci-vtt-qiskit")
            print(f'    export DEVICES=("{self._device}")')
            print(f"    srun --partition q_fiqci ... bash -c \"source $RUN_SETUP && python script.py\"")
            raise EnvironmentError(
                f"{url_env} not set. Are you running in the q_fiqci partition "
                f"with the fiqci-vtt-qiskit module loaded?"
            )

        try:
            from iqm.qiskit_iqm import IQMProvider
        except ImportError:
            raise ImportError(
                "IQM QPU backend requires 'qiskit-iqm' package.\n"
                "This is provided by the fiqci-vtt-qiskit module:\n"
                "  module use /appl/local/quantum/modulefiles\n"
                "  module load fiqci-vtt-qiskit"
            )

        provider = IQMProvider(url)
        self._sim = provider.get_backend()
        print(f"  Connected to {self._device}: {url[:40]}...")
        print(f"  Backend: {self._sim.name}")

        # Print qubit count if available
        try:
            n_qubits = self._sim.num_qubits
            print(f"  Qubits: {n_qubits}")
        except Exception:
            pass

    def run_circuits(self, jobs: list[CircuitJob]) -> list[CircuitResult]:
        """Submit circuits to Q50 and collect measurement results.

        Note: QPU returns measurement counts, not expectation values.
        Energy is computed from counts via expectation_from_counts().
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
        """Transpile circuit for IQM native gate set (CZ + phased-RX).

        IQM basis gates: CZ (entangling) and prx (phased rotation).
        The transpiler handles qubit routing for the Q50 topology.
        """
        self._ensure_sim()
        from qiskit import transpile
        return transpile(circuit, backend=self._sim, optimization_level=2)

    def capabilities(self) -> BackendCapabilities:
        max_q = 50 if self._device == "Q50" else 5
        return BackendCapabilities(
            max_qubits=max_q,
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
        max_q = 50 if self._device == "Q50" else 5
        if nq and nq > max_q:
            errors.append(f"IQM {self._device} supports max {max_q} qubits, got {nq}")
        return errors

    def estimate_walltime(self, config: ExperimentConfig) -> int:
        """QPU jobs are slow — budget generously."""
        maxiter = config.optimizer_params.get("maxiter", 100)
        shots = config.backend_params.get("shots", 1024)
        # ~2s per circuit execution on Q50, 2n circuits per gradient
        n_params = (config.num_qubits or 8) * 3
        sec_per_iter = 2 * 2 * n_params  # 2s × 2n circuits
        return max(600, sec_per_iter * maxiter * 2)  # 2× safety margin

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
