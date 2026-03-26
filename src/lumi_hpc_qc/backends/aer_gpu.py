# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Aer GPU backend — statevector and density matrix simulation on AMD MI250X.

Contains all tested knowledge from lumi_vqa:
  - decompose_for_aer(): multi-round decomposition to primitive gates
  - Precision: double by default, single configurable
  - Cache blocking for large qubit counts
  - save_expectation_value for energy evaluation
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

# Gates that qiskit-aer can execute directly
_AER_PRIMITIVE_GATES = {
    'x', 'y', 'z', 'h', 's', 'sdg', 't', 'tdg', 'sx', 'sxdg',
    'rx', 'ry', 'rz', 'rxx', 'ryy', 'rzz', 'cx', 'cy', 'cz',
    'cp', 'crx', 'cry', 'crz', 'ccx', 'csx', 'swap', 'cswap',
    'u', 'u1', 'u2', 'u3', 'p', 'id', 'ecr', 'measure', 'barrier',
    'save_expectation_value', 'save_statevector',
}


def decompose_for_aer(circuit, max_rounds=10):
    """Decompose circuit until all gates are Aer-compatible primitives.

    EfficientSU2, EvolvedOps (UCCSD), and other composite gates must be
    decomposed before Aer can execute them. A single decompose() only
    strips one layer of nesting — we need multiple rounds.
    """
    for round_i in range(max_rounds):
        needs_more = False
        for inst in circuit.data:
            if inst.operation.name not in _AER_PRIMITIVE_GATES:
                needs_more = True
                break
        if not needs_more:
            return circuit, round_i
        circuit = circuit.decompose()
    return circuit, max_rounds


class AerGpuBackend(Backend):
    """Qiskit Aer GPU backend — statevector and density matrix simulation."""

    name = "aer_gpu"

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config
        self._sim = None
        self._precision = "double"
        self._use_blocking = False
        self._blocking_qubits = 0

        if config:
            self._precision = config.precision
            bp = config.backend_params
            self._method = bp.get("method", "statevector")

            # Cache blocking for large qubit counts
            nq = config.num_qubits
            if self._precision == "double" and nq >= 30:
                self._use_blocking = True
                self._blocking_qubits = 29
            elif self._precision == "single" and nq >= 30:
                self._use_blocking = True
                self._blocking_qubits = 28
        else:
            self._method = "statevector"

    def _ensure_sim(self) -> None:
        """Lazily initialize AerSimulator (avoids import at module load)."""
        if self._sim is not None:
            return

        from qiskit_aer import AerSimulator

        self._sim = AerSimulator(
            method=self._method,
            device='GPU',
            precision=self._precision,
        )

    def run_circuits(self, jobs: list[CircuitJob]) -> list[CircuitResult]:
        import time
        self._ensure_sim()
        results = []

        for job in jobs:
            t0 = time.time()
            energies = []

            for i, circuit in enumerate(job.circuits):
                # Bind parameters if provided
                if job.parameters and i < len(job.parameters):
                    param_dict = job.parameters[i]
                    bound = circuit.assign_parameters(param_dict)
                else:
                    bound = circuit

                # Add expectation value measurement if observable provided
                if job.observable is not None:
                    bound.save_expectation_value(
                        job.observable,
                        list(range(circuit.num_qubits)),
                        label='energy',
                    )

                r = self._sim.run(
                    bound, shots=job.shots, seed_simulator=42,
                    blocking_enable=self._use_blocking,
                    blocking_qubits=self._blocking_qubits,
                ).result()

                if job.shots == 0 and job.observable is not None:
                    energy = float(np.real(r.data()['energy']))
                    energies.append(energy)

            elapsed = time.time() - t0
            results.append(CircuitResult(
                job_id=job.job_id,
                energies=energies if energies else None,
                execution_time_s=elapsed,
                backend_name=self.name,
            ))

        return results

    def compile_circuit(self, circuit):
        """Decompose to Aer-compatible primitive gates."""
        compiled, rounds = decompose_for_aer(circuit)
        if rounds > 0:
            print(f"  Decomposed to primitive gates in {rounds} round(s) "
                  f"→ {compiled.size()} gates, depth {compiled.depth()}")
        return compiled

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            max_qubits=44,
            supports_statevector=True,
            supports_density_matrix=True,
            supports_mps=False,
            supports_shots=True,
            requires_gpu=True,
            requires_cpu_only=False,
            slurm_partition="standard-g",
        )

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        errors = []
        nq = config.num_qubits
        if nq > 44:
            errors.append(f"Aer GPU supports max 44 qubits, got {nq}")
        if config.precision == "double" and nq > 36:
            errors.append(
                f"Double precision limited to 36 qubits on LUMI (got {nq}). "
                f"Set precision: single for {nq}q."
            )
        return errors

    def estimate_walltime(self, config: ExperimentConfig) -> int:
        nq = config.num_qubits or 12
        maxiter = config.optimizer_params.get("maxiter", 200)
        # Empirical from LUMI testing
        sec_per_eval = {12: 0.007, 18: 0.05, 24: 0.5, 30: 5, 36: 30, 40: 120, 44: 600}
        spe = sec_per_eval.get(nq, 0.007 * (2 ** (nq - 12)))
        n_params_est = nq * 3
        circuits_per_step = 2 * n_params_est + 1
        total = spe * circuits_per_step * maxiter * 2  # 2x safety
        return max(300, min(int(total), 48 * 3600))
