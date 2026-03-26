# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Aer CPU backend — matrix product state (MPS) and statevector on CPU.

MPS scales to larger qubit counts than statevector for low-entanglement
circuits, running on CPU without GPU requirement. Uses LUMI's 'standard'
partition instead of 'standard-g'.

Also supports CPU statevector for small systems where GPU overhead
exceeds the simulation time.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lumi_hpc_qc.backends.base import Backend
from lumi_hpc_qc.backends.aer_gpu import decompose_for_aer
from lumi_hpc_qc.types import (
    BackendCapabilities,
    CircuitJob,
    CircuitResult,
    ExperimentConfig,
)


class AerCpuBackend(Backend):
    """Qiskit Aer CPU backend — MPS and statevector on CPU cores."""

    name = "aer_cpu"

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config
        self._sim = None
        self._precision = "double"
        self._use_blocking = False
        self._blocking_qubits = 0

        if config:
            self._precision = config.precision
            bp = config.backend_params
            self._method = bp.get("method", "matrix_product_state")
            self._max_bond_dimension = bp.get("max_bond_dimension", 256)
        else:
            self._method = "matrix_product_state"
            self._max_bond_dimension = 256

    def _ensure_sim(self) -> None:
        """Lazily initialize AerSimulator for CPU."""
        if self._sim is not None:
            return

        from qiskit_aer import AerSimulator

        sim_opts = {
            "method": self._method,
            "device": "CPU",
            "precision": self._precision,
        }
        if self._method == "matrix_product_state":
            sim_opts["matrix_product_state_max_bond_dimension"] = self._max_bond_dimension

        self._sim = AerSimulator(**sim_opts)

    def run_circuits(self, jobs: list[CircuitJob]) -> list[CircuitResult]:
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

                if job.observable is not None:
                    bound.save_expectation_value(
                        job.observable,
                        list(range(circuit.num_qubits)),
                        label='energy',
                    )

                r = self._sim.run(bound, shots=job.shots, seed_simulator=42).result()

                if job.shots == 0 and job.observable is not None:
                    energies.append(float(np.real(r.data()['energy'])))

            elapsed = time.time() - t0
            results.append(CircuitResult(
                job_id=job.job_id,
                energies=energies if energies else None,
                execution_time_s=elapsed,
                backend_name=self.name,
            ))

        return results

    def compile_circuit(self, circuit):
        compiled, rounds = decompose_for_aer(circuit)
        if rounds > 0:
            print(f"  Decomposed to primitive gates in {rounds} round(s)")
        return compiled

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=100,  # MPS can handle large systems
            supports_statevector=True,
            supports_density_matrix=False,
            supports_mps=True,
            supports_shots=True,
            requires_gpu=False,
            requires_cpu_only=True,
            slurm_partition="standard",
        )

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        errors = []
        if self._method == "statevector" and (config.num_qubits or 0) > 30:
            errors.append(
                f"CPU statevector limited to ~30 qubits (got {config.num_qubits}). "
                f"Use method: matrix_product_state for larger systems."
            )
        return errors
