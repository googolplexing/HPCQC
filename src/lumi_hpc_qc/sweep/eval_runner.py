# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Evaluation-only runner — execute fixed circuits without an optimizer.

For non-parameterized circuits (BYO characterization circuits, reference
topology circuits), this module:
  1. Takes a fixed QuantumCircuit (no Parameters)
  2. Transpiles to the target placement's physical qubits
  3. Executes on the assigned backend (CPU or GPU via execution planner)
  4. Computes ⟨H⟩ from counts (if observable provided) or collects raw counts
  5. Returns a single result (not a trajectory)

This is the "evaluation-only mode" from RED-SPEC-002 §7.2.
VE15: Non-parameterized circuit runs in evaluation-only mode (no optimizer).

The key difference from VQE mode:
  - No optimizer loop
  - No gradient computation
  - No convergence tracking
  - Single execution (or repeat for statistical averaging)
  - Result is a single energy value, not a trajectory
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from lumi_hpc_qc.sweep.circuit_loader import LoadedCircuit


@dataclass
class EvalResult:
    """Result of a single evaluation-only execution.

    Attributes:
        energy: Expectation value of the observable (None if no observable).
        counts: Raw measurement counts (None if shots=0).
        execution_time_s: Wall time for the execution.
        num_shots: Shots used (0 for exact methods).
        transpiled_depth: Circuit depth after transpilation.
        transpiled_cx_count: Number of CX/CZ gates after transpilation.
        backend_used: Which backend executed this ("aer_cpu", "aer_gpu", etc.).
        error: Error message if execution failed (None on success).
        metadata: Additional metadata from the execution.
    """
    energy: float | None = None
    counts: dict[str, int] | None = None
    execution_time_s: float = 0.0
    num_shots: int = 0
    transpiled_depth: int = 0
    transpiled_cx_count: int = 0
    backend_used: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def evaluate_circuit(
    loaded: LoadedCircuit,
    *,
    observable: Any | None = None,
    method: str = "density_matrix",
    shots: int = 4096,
    seed: int = 42,
    noise_model: Any | None = None,
    coupling_map: Any | None = None,
    initial_layout: list[int] | None = None,
    device: str = "CPU",
) -> EvalResult:
    """Execute a fixed circuit and return the result.

    For exact methods (shots=0), computes ⟨H⟩ via trace(H @ ρ).
    For shot-based methods, estimates ⟨H⟩ from measurement counts.
    If no observable is provided, returns only raw counts.

    Args:
        loaded: A LoadedCircuit from the circuit loader.
        observable: SparsePauliOp for expectation value computation.
        method: "density_matrix" or "statevector".
        shots: Number of measurement shots (0 for exact).
        seed: Simulator seed for reproducibility.
        noise_model: Qiskit Aer NoiseModel (None = noiseless).
        coupling_map: CouplingMap for transpilation (None = all-to-all).
        initial_layout: Physical qubit mapping for transpilation.
        device: "CPU" or "GPU" for AerSimulator device selection.

    Returns:
        EvalResult with energy and/or counts.

    Raises:
        ValueError: If circuit is parameterized (has unbound Parameters).
    """
    if loaded.is_parameterized:
        raise ValueError(
            f"Cannot run eval-only on parameterized circuit "
            f"({loaded.num_parameters} unbound parameters). "
            f"Use VQE workflow for parameterized circuits."
        )

    from qiskit_aer import AerSimulator
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    t0 = time.time()
    result = EvalResult(backend_used=f"aer_{device.lower()}")

    try:
        # Build simulator
        sim_options = {"method": method, "device": device}
        if noise_model is not None:
            sim_options["noise_model"] = noise_model
        sim = AerSimulator(**sim_options)

        # Transpile to target topology
        qc = loaded.circuit.copy()

        if coupling_map is not None or initial_layout is not None:
            pm = generate_preset_pass_manager(
                optimization_level=1,
                backend=sim,
                initial_layout=initial_layout,
            )
            qc = pm.run(qc)

        result.transpiled_depth = qc.depth()
        result.transpiled_cx_count = sum(
            1 for inst in qc.data
            if inst.operation.num_qubits == 2
        )

        # Execute
        if method == "statevector" or shots == 0:
            # Exact evaluation via density matrix trace
            qc_exec = qc.copy()
            qc_exec.save_density_matrix()
            job = sim.run(qc_exec, shots=0, seed_simulator=seed)
            sim_result = job.result()
            dm = sim_result.data()["density_matrix"]

            if observable is not None:
                h_matrix = observable.to_matrix()
                dm_array = np.array(dm)
                result.energy = float(np.real(np.trace(h_matrix @ dm_array)))
            result.num_shots = 0

        else:
            # Shot-based evaluation
            qc_exec = qc.copy()
            # Remove any save_* instructions
            qc_exec.measure_all()

            job = sim.run(qc_exec, shots=shots, seed_simulator=seed)
            sim_result = job.result()
            result.counts = sim_result.get_counts()
            result.num_shots = shots

            if observable is not None:
                result.energy = _energy_from_counts(
                    result.counts, observable, loaded.num_qubits
                )

    except Exception as e:
        result.error = str(e)

    result.execution_time_s = time.time() - t0
    return result


def evaluate_batch(
    loaded: LoadedCircuit,
    configs: list[dict[str, Any]],
    *,
    max_workers: int | None = None,
) -> list[EvalResult]:
    """Execute the same circuit with multiple configurations.

    Each config dict is passed as kwargs to evaluate_circuit().
    Useful for running one reference circuit across multiple noise
    environments or seeds.

    Args:
        loaded: The fixed circuit to evaluate.
        configs: List of config dicts, each passed to evaluate_circuit().
        max_workers: Number of parallel workers (None = sequential).

    Returns:
        List of EvalResult, one per config, in order.
    """
    if max_workers is None or max_workers <= 1:
        return [evaluate_circuit(loaded, **cfg) for cfg in configs]

    # Parallel execution via multiprocessing
    # Use Pool(1) pattern to avoid fork-after-threading
    import multiprocessing as mp

    def _worker(args):
        idx, cfg = args
        r = evaluate_circuit(loaded, **cfg)
        return (idx, r)

    with mp.Pool(max_workers) as pool:
        indexed = [(i, cfg) for i, cfg in enumerate(configs)]
        raw = pool.map(_worker, indexed)

    # Restore original order
    raw.sort(key=lambda x: x[0])
    return [r for _, r in raw]


def _energy_from_counts(
    counts: dict[str, int],
    observable: Any,
    num_qubits: int,
) -> float:
    """Estimate ⟨H⟩ from measurement counts.

    Simple implementation: for each Pauli term, compute the expectation
    from the measurement counts by evaluating the parity of the relevant
    qubits. This is a simplified version — the full implementation in
    pauli_measurement.py handles grouped measurements.
    """
    total_shots = sum(counts.values())
    if total_shots == 0:
        return 0.0

    energy = 0.0
    for pauli_label, coeff in zip(observable.paulis.to_labels(),
                                   observable.coeffs):
        # For each Pauli string, compute expectation from counts
        parity_sum = 0.0
        for bitstring, count in counts.items():
            # Reverse bitstring (qiskit convention: q[0] is rightmost)
            bits = bitstring[::-1]
            parity = 0
            for i, p in enumerate(pauli_label[::-1]):
                if p in ("Z", "Y") and i < len(bits):
                    if bits[i] == "1":
                        parity ^= 1
            sign = 1 - 2 * parity  # +1 for even parity, -1 for odd
            parity_sum += sign * count
        energy += float(np.real(coeff)) * (parity_sum / total_shots)

    return energy
