# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""IQM QPU backend — VTT Q50 quantum processor via FiQCI middleware.

Fixes applied:
  F1: Shot-based evaluation uses basis-rotated measurement circuits
      for correct handling of X/Y Pauli terms.
  Q2: Fixed observable.to_list() dual coefficient source.

Submits circuits to the IQM Q50 quantum processor through CSC's
HPC-Quantum middleware on LUMI. No API token required — access is
controlled through SLURM's q_fiqci partition and FiQCI modules.

Access pattern (from CSC docs):
    module use /appl/local/quantum/modulefiles
    module load fiqci-vtt-qiskit
    export DEVICES=("Q50")
    srun --partition q_fiqci -c 1 -n 1 bash -c "source $RUN_SETUP && python script.py"

The Q50_CORTEX_URL environment variable is set automatically by the
FiQCI module when running inside the q_fiqci partition.

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
    """IQM Q50 quantum processor backend via FiQCI."""

    name = "iqm_qpu"

    # VTT enforces max 200 circuits per batch (QX FAQ)
    VTT_BATCH_LIMIT = 200

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config
        self._sim = None  # IQM backend object (named _sim for interface compat)
        self._device = "Q50"
        self._shots = 4096
        self._calibration_set_id = None

        if config:
            bp = config.backend_params
            self._device = bp.get("device", "Q50")
            self._shots = bp.get("shots", 4096)
            self._calibration_set_id = bp.get("calibration_set_id")

    def _ensure_sim(self) -> None:
        """Connect to Q50 via IQM provider."""
        if self._sim is not None:
            return

        if self._device == "Q50":
            url_env = "Q50_CORTEX_URL"
        else:
            url_env = "HELMI_CORTEX_URL"

        url = os.getenv(url_env, "")

        if not url:
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

        try:
            n_qubits = self._sim.num_qubits
            print(f"  Qubits: {n_qubits}")
        except Exception:
            pass

    def run_circuits(self, jobs: list[CircuitJob]) -> list[CircuitResult]:
        """Submit circuits to Q50 and collect measurement results.

        F1 FIX: Uses basis-rotated measurement circuits for correct
        handling of X/Y Pauli terms. Each commuting group of Pauli
        terms gets its own measurement circuit with appropriate
        rotation gates (H for X, Sdg+H for Y) before Z-basis measurement.
        """
        import time
        from qiskit import transpile

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

                shots = job.shots if job.shots > 0 else self._shots

                if job.observable is not None:
                    # F1 FIX: build basis-rotated measurement circuits
                    from lumi_hpc_qc.backends.pauli_measurement import (
                        build_measurement_circuits,
                        expectation_from_grouped_counts,
                    )

                    meas_circuits, meas_groups, identity_e = (
                        build_measurement_circuits(bound, job.observable, shots)
                    )

                    # Transpile each circuit for Q50 native gates
                    try:
                        from iqm.qiskit_iqm.iqm_transpilation import optimize_single_qubit_gates
                        transpiled = []
                        for mc in meas_circuits:
                            t = transpile(mc, backend=self._sim, optimization_level=3)
                            t = optimize_single_qubit_gates(t)
                            transpiled.append(t)
                        meas_circuits = transpiled
                    except ImportError:
                        meas_circuits = transpile(
                            meas_circuits, backend=self._sim, optimization_level=2
                        )

                    # ── Batch submission to QPU ──
                    # IQM .run() accepts list[QuantumCircuit] as single batch.
                    # VTT caps at 200 circuits/batch — auto-chunk if needed.
                    counts_list = self._submit_batch(
                        meas_circuits, shots=shots,
                    )

                    energy = expectation_from_grouped_counts(
                        counts_list, meas_groups, identity_e, shots
                    )
                    energies.append(energy)

                else:
                    # No observable — just run and collect counts
                    if not any(inst.operation.name == 'measure' for inst in bound.data):
                        bound.measure_all()
                    qpu_result = self._sim.run(bound, shots=shots).result()
                    counts = qpu_result.get_counts()

            elapsed = time.time() - t0
            results.append(CircuitResult(
                job_id=job.job_id,
                energies=energies if energies else None,
                counts=counts if not energies else None,
                execution_time_s=elapsed,
                backend_name=self.name,
            ))

        return results

    def _submit_batch(
        self,
        circuits: list,
        shots: int,
    ) -> list[dict[str, int]]:
        """Submit circuits as batched .run() calls, auto-chunking at VTT limit.

        IQM's IQMBackend.run() accepts list[QuantumCircuit] as a single
        batch job — one queue entry for the entire list. VTT enforces a
        hard cap of 200 circuits per batch (QX FAQ). This method:

        1. If len(circuits) <= VTT_BATCH_LIMIT: single .run() call
        2. If len(circuits) > VTT_BATCH_LIMIT: chunk into sequential
           batches of <=200, submit each, reassemble in original order

        Returns:
            List of count dicts in the same order as input circuits.
            result[i] corresponds to circuits[i].

        Ordering guarantee: IQM .run(list) returns results in submission
        order — result.get_counts(i) corresponds to circuit_list[i].
        Verified by Test A (single batch) and Test B (multi-batch).
        """
        if not circuits:
            return []

        all_counts: list[dict[str, int]] = []
        limit = self.VTT_BATCH_LIMIT

        for chunk_start in range(0, len(circuits), limit):
            chunk = circuits[chunk_start:chunk_start + limit]

            if len(chunk) == 1:
                # Single circuit — submit directly (no list wrapping needed)
                result = self._sim.run(chunk[0], shots=shots).result()
                all_counts.append(result.get_counts())
            else:
                # Batch submission — one queue wait for the entire chunk
                result = self._sim.run(chunk, shots=shots).result()
                for i in range(len(chunk)):
                    all_counts.append(result.get_counts(i))

        return all_counts

    def compile_circuit(self, circuit):
        """Transpile circuit for IQM native gate set (CZ + phased-RX)."""
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
        maxiter = config.optimizer_params.get("maxiter", 100)
        shots = config.backend_params.get("shots", 1024)
        n_params = (config.num_qubits or 8) * 3
        sec_per_iter = 2 * 2 * n_params
        return max(600, sec_per_iter * maxiter * 2)
