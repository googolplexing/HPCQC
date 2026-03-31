# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Aer GPU backend — statevector and density matrix simulation on AMD MI250X.

Fixes applied:
  F1: Shot-based evaluation uses basis-rotated measurement circuits
      (pauli_measurement module) instead of broken Z-only parity.
  C1: Noisy simulation transpiles circuits to Q50 coupling map,
      matching real QPU routing overhead.

Contains all tested knowledge from lumi_vqa:
  - decompose_for_aer(): multi-round decomposition to primitive gates
  - Precision: double by default, single configurable
  - Cache blocking for large qubit counts
  - save_expectation_value for noiseless energy evaluation
  - Noise model support for Q50 benchmarking (Phase 3)
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
        self._noise_model = None
        self._noise_model_file = None
        self._coupling_map = None
        self._shots = 0

        # Phase B: decoupled coupling map and noise channels
        self._coupling_map_source = "full"
        self._coupling_map_file = None
        self._noise_channels = None  # None = all active (backward compatible)

        if config:
            self._precision = config.precision
            bp = config.backend_params
            self._method = bp.get("method", "statevector")
            self._noise_model_file = bp.get("noise_model_file", None)
            self._shots = bp.get("shots", 0)
            self._coupling_map_source = bp.get("coupling_map_source", "full")
            self._coupling_map_file = bp.get("coupling_map_file",
                                              bp.get("noise_model_file", None))
            self._noise_channels = bp.get("noise_channels", None)

            # Cache blocking for large qubit counts
            nq = config.num_qubits
            if nq and self._precision == "double" and nq >= 30:
                self._use_blocking = True
                self._blocking_qubits = 29
            elif nq and self._precision == "single" and nq >= 30:
                self._use_blocking = True
                self._blocking_qubits = 28
        else:
            self._method = "statevector"

    def _ensure_sim(self) -> None:
        """Lazily initialize AerSimulator (avoids import at module load)."""
        if self._sim is not None:
            return

        from qiskit_aer import AerSimulator
        import os

        num_qubits = self._config.num_qubits if self._config else 8

        # Resolve calibration file path
        cal_path = self._coupling_map_file or self._noise_model_file
        if cal_path and not os.path.isabs(cal_path):
            project_dir = os.environ.get("PROJECT_DIR",
                          os.environ.get("SINGULARITYENV_PROJECT_DIR", "."))
            cal_path = os.path.join(project_dir, cal_path)

        # ── Load coupling map independently of noise model ──
        if self._coupling_map_source == "calibration" and cal_path:
            from lumi_hpc_qc.backends.noise_model import extract_coupling_map
            self._coupling_map = extract_coupling_map(cal_path, num_qubits)

        # ── Build noise model if requested ──
        if self._noise_model_file:
            from lumi_hpc_qc.backends.noise_model import build_noise_model

            nm_path = self._noise_model_file
            if not os.path.isabs(nm_path):
                project_dir = os.environ.get("PROJECT_DIR",
                              os.environ.get("SINGULARITYENV_PROJECT_DIR", "."))
                nm_path = os.path.join(project_dir, nm_path)

            self._noise_model, nm_coupling_map = build_noise_model(
                nm_path, num_qubits, noise_channels=self._noise_channels
            )
            if self._coupling_map is None:
                self._coupling_map = nm_coupling_map
            print(f"  Noise model loaded from: {self._noise_model_file}")

        # Determine device
        device = 'GPU'

        sim_kwargs = dict(
            method=self._method,
            device=device,
            precision=self._precision,
        )
        if self._noise_model is not None:
            sim_kwargs["noise_model"] = self._noise_model

        self._sim = AerSimulator(**sim_kwargs)

    def run_circuits(self, jobs: list[CircuitJob]) -> list[CircuitResult]:
        """Execute circuit jobs on the Aer simulator.

        Handles two modes:
        - shots=0: exact statevector via save_expectation_value (noiseless)
        - shots>0: basis-rotated measurement circuits (F1 fix) with
                   optional coupling map transpilation (C1 fix)
        """
        import time
        self._ensure_sim()
        results = []

        for job in jobs:
            t0 = time.time()
            energies = []

            # Determine shots: job-level overrides instance default
            shots = job.shots if job.shots > 0 else self._shots

            if shots == 0 and job.observable is not None:
                # ── Noiseless statevector mode ──
                # Use save_expectation_value for exact Tr(ρH)
                for i, circuit in enumerate(job.circuits):
                    if job.parameters and i < len(job.parameters):
                        bound = circuit.assign_parameters(job.parameters[i])
                    else:
                        bound = circuit

                    bound.save_expectation_value(
                        job.observable,
                        list(range(circuit.num_qubits)),
                        label='energy',
                    )
                    r = self._sim.run(
                        bound, shots=0, seed_simulator=42,
                        blocking_enable=self._use_blocking,
                        blocking_qubits=self._blocking_qubits,
                    ).result()
                    energy = float(np.real(r.data()['energy']))
                    energies.append(energy)

            elif shots > 0 and job.observable is not None:
                # ── Shot-based mode (noisy simulation) ──
                # F1 FIX: use basis-rotated measurement circuits
                from lumi_hpc_qc.backends.pauli_measurement import (
                    build_measurement_circuits,
                    expectation_from_grouped_counts,
                )
                from qiskit import transpile

                for i, circuit in enumerate(job.circuits):
                    if job.parameters and i < len(job.parameters):
                        bound = circuit.assign_parameters(job.parameters[i])
                    else:
                        bound = circuit

                    # Build basis-rotated circuits for each Pauli group
                    meas_circuits, meas_groups, identity_e = (
                        build_measurement_circuits(bound, job.observable, shots)
                    )

                    # C1 FIX: transpile to coupling map if noise model active
                    # Qiskit 2.3.0 bug workaround: when a noise model is loaded
                    # into AerSimulator, the transpiler tries to introspect the
                    # simulator's _FakeTarget and hits a renamed attribute
                    # (_coupling_map → _coupling_graph). Passing basis_gates
                    # explicitly bypasses the _FakeTarget lookup entirely.
                    if self._coupling_map is not None:
                        basis_gates = (
                            self._noise_model.basis_gates
                            if self._noise_model is not None
                            else ['cx', 'cz', 'id', 'rz', 'sx', 'x',
                                  'reset', 'measure']
                        )
                        meas_circuits = transpile(
                            meas_circuits,
                            coupling_map=self._coupling_map,
                            basis_gates=basis_gates,
                            optimization_level=2,
                        )

                    # Run all measurement circuits
                    # C6 FIX: unique seed per circuit for realistic shot noise
                    counts_list = []
                    for ci, mc in enumerate(meas_circuits):
                        seed = 42 + hash((i, ci)) % (2**31)
                        r = self._sim.run(
                            mc, shots=shots, seed_simulator=seed,
                            blocking_enable=self._use_blocking,
                            blocking_qubits=self._blocking_qubits,
                        ).result()
                        counts_list.append(r.get_counts())

                    energy = expectation_from_grouped_counts(
                        counts_list, meas_groups, identity_e, shots
                    )
                    energies.append(energy)

            else:
                # No observable — just run circuits (measurement already added)
                for i, circuit in enumerate(job.circuits):
                    if job.parameters and i < len(job.parameters):
                        bound = circuit.assign_parameters(job.parameters[i])
                    else:
                        bound = circuit
                    self._sim.run(
                        bound, shots=shots or 1024,
                        blocking_enable=self._use_blocking,
                        blocking_qubits=self._blocking_qubits,
                    ).result()

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
        if nq and nq > 44:
            errors.append(f"Aer GPU supports max 44 qubits, got {nq}")
        if nq and config.precision == "double" and nq > 36:
            errors.append(
                f"Double precision limited to 36 qubits on LUMI (got {nq}). "
                f"Set precision: single for {nq}q."
            )
        bp = config.backend_params
        # Noise model with active channels requires density_matrix.
        # But coupling_map + statevector is valid (topology-noiseless mode).
        if bp.get("noise_model_file") and bp.get("method") == "statevector":
            nc = bp.get("noise_channels")
            if nc is None or any(nc.values()):
                errors.append(
                    "Noise model with active channels requires method: density_matrix. "
                    "For topology-only, use coupling_map_source: calibration "
                    "without noise_model_file."
                )
        return errors

    def estimate_walltime(self, config: ExperimentConfig) -> int:
        nq = config.num_qubits or 12
        maxiter = config.optimizer_params.get("maxiter", 200)
        sec_per_eval = {12: 0.007, 18: 0.05, 24: 0.5, 30: 5, 36: 30, 40: 120, 44: 600}
        spe = sec_per_eval.get(nq, 0.007 * (2 ** (nq - 12)))
        if self._noise_model is not None:
            spe *= 4
        n_params_est = nq * 3
        circuits_per_step = 2 * n_params_est + 1
        total = spe * circuits_per_step * maxiter * 2
        return max(300, min(int(total), 48 * 3600))
