# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""IQM QPU backend — VTT Q50 quantum processor via FiQCI middleware.

Fixes applied:
  F1: Shot-based evaluation uses basis-rotated measurement circuits
      for correct handling of X/Y Pauli terms.
  Q2: Fixed observable.to_list() dual coefficient source.

v1.3.0 additions (RED-DIRECTIVE-V130-v1.0):
  Item 2: QXClient integration — capture_job_timing() for server-side
          QPU timing from VTT QX API timeline.
  Item 3: Dynamic batch limit from get_job_policy() at connection time.
  Item 7: Batch retry with exponential backoff (3 retries, 1s/2s/4s).

v1.3.1 changes (RED-DIRECTIVE-QPU-CONFIG-v1.0):
  - Retry DISABLED by default.  Configurable via QPUConfig / YAML.
  - Error classification: only retryable_errors patterns trigger retry.
  - QXClient / timing capture opt-in via QPUConfig.timing_capture.
  - Queue prefetch opt-in via QPUConfig.queue_prefetch.
  - Connection timeout via signal.alarm (Finding 5).
  - Auto-chunk logging when batches exceed VTT limit (Finding 3).
  - Shots read from QPUConfig single source (Finding 6).
  - retry_attempts tracked per batch for benchmark Parquet (Finding 7).

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
import signal
import time
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

    # VTT default batch limit — overridden at connection time by
    # get_job_policy() if the QX API is reachable and QPUConfig allows.
    VTT_BATCH_LIMIT = 100

    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self._config = config
        self._sim = None  # IQM backend object (named _sim for interface compat)
        self._qx = None   # QXClient for timing + monitoring (opt-in)
        self._device = "Q50"
        self._calibration_set_id = None

        # QPU timing accumulator — one QPUJobTiming per _submit_batch() call.
        # Fed into benchmark Parquet export at sweep completion.
        self._batch_timings: list = []

        # Retry attempts per batch — parallel to _batch_timings.
        # 1 = first-try success, 2+ = retried. (Finding 7)
        self._batch_retry_attempts: list[int] = []

        # Queue length captured once before a sweep if queue_prefetch enabled.
        self._queue_length_before: int | None = None

        # ── QPUConfig defaults: retry OFF, timing OFF, queue OFF ──
        self._retry_enabled: bool = False
        self._retry_max_attempts: int = 3
        self._retry_base_wait_s: float = 1.0
        self._retry_errors: list[str] = [
            "No results available", "timeout", "ConnectionError",
        ]
        self._timing_capture: bool = False
        self._queue_prefetch: bool = False
        self._connection_timeout_s: int = 60
        self._shots: int = 4096

        if config:
            bp = config.backend_params
            self._device = bp.get("device", "Q50")
            self._calibration_set_id = bp.get("calibration_set_id")

            # Read QPUConfig fields if passed through backend_params
            qpu = bp.get("qpu_config")
            if qpu is not None:
                self._apply_qpu_config(qpu)
            else:
                self._shots = bp.get("shots", 4096)

    def _apply_qpu_config(self, qpu) -> None:
        """Apply QPUConfig (dataclass or duck-typed object) to instance."""
        self._shots = getattr(qpu, "shots", 4096)
        self._connection_timeout_s = getattr(qpu, "connection_timeout_s", 60)
        self._retry_enabled = getattr(qpu, "retry_enabled", False)
        self._retry_max_attempts = getattr(qpu, "retry_max_attempts", 3)
        self._retry_base_wait_s = getattr(qpu, "retry_base_wait_s", 1.0)
        self._retry_errors = getattr(qpu, "retry_errors", self._retry_errors)
        self._timing_capture = getattr(qpu, "timing_capture", False)
        self._queue_prefetch = getattr(qpu, "queue_prefetch", False)
        bl = getattr(qpu, "batch_limit", None)
        if bl is not None:
            self.VTT_BATCH_LIMIT = bl

    def set_qpu_config(self, qpu_config) -> None:
        """Apply QPUConfig after construction.

        Called by the sweep engine when it has parsed the YAML qpu:
        section but the backend was already instantiated.
        """
        self._apply_qpu_config(qpu_config)

    def _ensure_sim(self) -> None:
        """Connect to Q50 via IQM provider, optionally create QXClient."""
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

        # ── Finding 5: Connection timeout via signal.alarm ──
        timeout_s = self._connection_timeout_s

        def _timeout_handler(signum, frame):
            raise TimeoutError(
                f"{self._device} did not respond within {timeout_s}s. "
                f"The device may be offline. VTT calibrates Q50 at 06:30 and 19:00 UTC. "
                f"Check device status or increase qpu.connection_timeout_s in your campaign YAML."
            )

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
        try:
            provider = IQMProvider(url)
            self._sim = provider.get_backend()
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        print(f"  Connected to {self._device}: {url[:40]}...")
        print(f"  Backend: {self._sim.name}")

        try:
            n_qubits = self._sim.num_qubits
            print(f"  Qubits: {n_qubits}")
        except Exception:
            pass

        # ── Print QPU config summary ──
        retry_status = "ENABLED" if self._retry_enabled else "DISABLED"
        print(f"  QPU config: retry={retry_status}, timing_capture={self._timing_capture}, "
              f"queue_prefetch={self._queue_prefetch}, shots={self._shots}")

        # ── QXClient: only if timing_capture is enabled (Finding 4) ──
        if self._timing_capture:
            try:
                from lumi_hpc_qc.backends.qx_client import QXClient
                self._qx = QXClient.from_backend(self._sim)
                print(f"  QXClient: initialized (timing capture enabled)")
            except Exception as e:
                self._qx = None
                print(f"  QXClient: unavailable ({e}) — timing capture disabled")

        # ── Dynamic batch limit: only if not overridden by config ──
        if self._qx is not None and self.VTT_BATCH_LIMIT == 100:
            try:
                policy = self._qx.get_job_policy()
                if policy and "max_number_circuits_per_batch" in policy:
                    new_limit = policy["max_number_circuits_per_batch"]
                    print(f"  Batch limit: {new_limit} (from QX API job policy)")
                    self.VTT_BATCH_LIMIT = new_limit
                else:
                    print(f"  Batch limit: {self.VTT_BATCH_LIMIT} (default — policy endpoint empty)")
            except Exception:
                print(f"  Batch limit: {self.VTT_BATCH_LIMIT} (default — policy endpoint unreachable)")
        else:
            print(f"  Batch limit: {self.VTT_BATCH_LIMIT}")

        # ── Queue length: only if queue_prefetch enabled (Finding 4) ──
        if self._queue_prefetch and self._qx is not None:
            try:
                self._queue_length_before = self._qx.get_queue_length()
                if self._queue_length_before is not None:
                    print(f"  Queue length: {self._queue_length_before}")
            except Exception:
                pass

    def run_circuits(self, jobs: list[CircuitJob]) -> list[CircuitResult]:
        """Submit circuits to Q50 and collect measurement results.

        F1 FIX: Uses basis-rotated measurement circuits for correct
        handling of X/Y Pauli terms. Each commuting group of Pauli
        terms gets its own measurement circuit with appropriate
        rotation gates (H for X, Sdg+H for Y) before Z-basis measurement.
        """
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

        v1.3.1 changes (RED-DIRECTIVE-QPU-CONFIG-v1.0):
          - Auto-chunk logging: clear message when splitting (Finding 3).
          - Retry disabled by default; error-classified when enabled (Finding 1).
          - retry_attempts tracked per chunk for benchmark Parquet (Finding 7).

        Returns:
            List of count dicts in the same order as input circuits.
        """
        if not circuits:
            return []

        all_counts: list[dict[str, int]] = []
        limit = self.VTT_BATCH_LIMIT

        # ── Finding 3: Auto-chunk logging ──
        if len(circuits) > limit:
            n_chunks = -(-len(circuits) // limit)  # ceil division
            chunk_sizes = []
            for i in range(0, len(circuits), limit):
                chunk_sizes.append(min(limit, len(circuits) - i))
            print(f"  \u26a0 Batch of {len(circuits)} circuits exceeds VTT limit ({limit}).")
            print(f"    Splitting into {n_chunks} chunks: {chunk_sizes}")
            print(f"    Each chunk incurs a separate queue wait (~20s typical).")

        for chunk_start in range(0, len(circuits), limit):
            chunk = circuits[chunk_start:chunk_start + limit]
            chunk_for_run = chunk[0] if len(chunk) == 1 else chunk

            # ── Retry: OFF by default, error-classified when ON (Finding 1) ──
            max_attempts = self._retry_max_attempts if self._retry_enabled else 1
            result = None
            timing = None
            attempts_taken = 0

            for attempt in range(max_attempts):
                attempts_taken = attempt + 1
                try:
                    if self._qx is not None:
                        qiskit_result, timing = self._qx.capture_job_timing(
                            self._sim, chunk_for_run, shots,
                            queue_length_before=self._queue_length_before,
                        )
                        result = qiskit_result
                    else:
                        result = self._sim.run(chunk_for_run, shots=shots).result()
                    break  # success
                except Exception as e:
                    if not self._retry_enabled:
                        raise  # retry disabled — fail immediately

                    # Check if this error is retryable
                    error_str = str(e)
                    is_retryable = any(
                        pattern in error_str
                        for pattern in self._retry_errors
                    )
                    if not is_retryable or attempt == max_attempts - 1:
                        print(f"  BATCH FAILED after {attempts_taken} attempt(s): {e}")
                        raise  # fatal error or last attempt

                    wait = self._retry_base_wait_s * (2 ** attempt)
                    print(f"  Retryable error (attempt {attempts_taken}/{max_attempts}): {e}")
                    print(f"  Retrying in {wait}s...")
                    time.sleep(wait)

            # Accumulate timing record (best-effort)
            if timing is not None:
                self._batch_timings.append(timing)

            # Track retry attempts for benchmark Parquet (Finding 7)
            self._batch_retry_attempts.append(attempts_taken)

            # Extract counts from result
            if len(chunk) == 1:
                all_counts.append(result.get_counts())
            else:
                for i in range(len(chunk)):
                    all_counts.append(result.get_counts(i))

        return all_counts

    def get_batch_timings(self) -> list:
        """Return accumulated QPUJobTiming records for benchmark export."""
        return list(self._batch_timings)

    def get_batch_retry_attempts(self) -> list[int]:
        """Return retry attempt counts per batch for benchmark Parquet.

        Parallel to get_batch_timings(). Value 1 = first-try success,
        2+ = retried. (RED-DIRECTIVE-QPU-CONFIG Finding 7)
        """
        return list(self._batch_retry_attempts)

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
