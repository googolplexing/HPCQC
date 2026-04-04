#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""QPU timing benchmark — per-circuit and per-batch overhead on Q50.

RED-RESP-PACKING-v1.0 §2.7

Measures:
  1. Wall time for batch of 10 circuits (SU2 reps=2, 4q, 4096 shots)
  2. Wall time for batch of 1 circuit (same)
  3. Derives: per_circuit_ms, per_batch_overhead_ms

Run via SLURM on q_fiqci partition:
    sbatch tests/test_v111_qpu_timing.sh
"""

import os
import sys
import time
import json

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))


def build_test_circuits(n_circuits: int, n_qubits: int = 4):
    """Build n_circuits identical SU2 reps=2 circuits for timing."""
    from qiskit.circuit.library import EfficientSU2
    from qiskit import transpile
    import numpy as np

    ansatz = EfficientSU2(n_qubits, reps=2, entanglement="linear")
    rng = np.random.default_rng(42)
    params = rng.uniform(0, 2 * np.pi, ansatz.num_parameters)
    bound = ansatz.assign_parameters(params)
    bound.measure_all()

    # Get backend for transpilation
    from iqm.qiskit_iqm import IQMProvider
    url = os.environ.get("Q50_CORTEX_URL", "")
    if not url:
        raise EnvironmentError("Q50_CORTEX_URL not set")

    provider = IQMProvider(url)
    backend = provider.get_backend()
    print(f"  Connected to Q50: {backend.num_qubits} qubits")

    # Transpile for Q50 native gates
    transpiled = transpile(bound, backend=backend, optimization_level=2)
    print(f"  Circuit depth: {transpiled.depth()}")
    print(f"  Gate count: {sum(transpiled.count_ops().values())}")

    # Create n_circuits copies (same circuit, same params)
    circuits = [transpiled.copy() for _ in range(n_circuits)]
    return circuits, backend


def time_batch(backend, circuits, shots=4096):
    """Submit batch and measure wall time."""
    t0 = time.time()
    if len(circuits) == 1:
        job = backend.run(circuits[0], shots=shots)
    else:
        job = backend.run(circuits, shots=shots)
    result = job.result()
    elapsed = time.time() - t0
    return elapsed, result


def main():
    print("\n═══ QPU TIMING BENCHMARK ═══\n")

    SHOTS = 4096
    N_BATCH = 10

    # Build circuits
    print(f"Building {N_BATCH + 1} circuits (SU2 reps=2, 4q, {SHOTS} shots)...")
    circuits, backend = build_test_circuits(N_BATCH + 1)

    # ── Benchmark 1: Batch of 10 ──
    print(f"\n── Benchmark 1: Batch of {N_BATCH} circuits ──")
    batch_10 = circuits[:N_BATCH]
    t_batch10, r10 = time_batch(backend, batch_10, SHOTS)
    print(f"  Wall time: {t_batch10:.3f}s")

    # ── Benchmark 2: Batch of 1 ──
    print(f"\n── Benchmark 2: Batch of 1 circuit ──")
    batch_1 = [circuits[N_BATCH]]
    t_batch1, r1 = time_batch(backend, batch_1, SHOTS)
    print(f"  Wall time: {t_batch1:.3f}s")

    # ── Derive metrics ──
    per_circuit_s = (t_batch10 - t_batch1) / (N_BATCH - 1)
    per_batch_overhead_s = t_batch1 - per_circuit_s
    per_circuit_ms = per_circuit_s * 1000
    per_batch_overhead_ms = per_batch_overhead_s * 1000

    # Infer reset method
    if per_circuit_ms < 50:
        inferred_reset = "active_reset (< 50ms/circuit)"
    elif per_circuit_ms < 200:
        inferred_reset = "fast_passive_or_active (50-200ms/circuit)"
    else:
        inferred_reset = "passive_reset (> 200ms/circuit)"

    print(f"\n═══ RESULTS ═══")
    print(f"  Batch of {N_BATCH}:     {t_batch10*1000:.1f} ms")
    print(f"  Batch of 1:       {t_batch1*1000:.1f} ms")
    print(f"  Per-circuit:      {per_circuit_ms:.1f} ms")
    print(f"  Per-batch overhead: {per_batch_overhead_ms:.1f} ms")
    print(f"  Inferred reset:   {inferred_reset}")
    print()

    # ── Campaign cost projections ──
    print("═══ CAMPAIGN PROJECTIONS (4q star, 108 placements, DSatur 16 rounds) ═══")
    circuits_per_iter = 32  # 16 rounds × 2 Pauli
    iter_time_ms = circuits_per_iter * per_circuit_ms + per_batch_overhead_ms
    campaign_400_s = 400 * iter_time_ms / 1000
    print(f"  Per VQE iteration: {iter_time_ms:.0f} ms ({circuits_per_iter} circuits)")
    print(f"  400-iteration campaign: {campaign_400_s:.1f}s ({campaign_400_s/60:.1f} min)")
    print()

    # Characterization mode (all circuits independent)
    total_circuits = 12800  # 108 placements × 2 groups × ~59 (some dedup)
    n_batches = (total_circuits + 199) // 200
    char_time_s = total_circuits * per_circuit_s + n_batches * per_batch_overhead_s
    print(f"  Full characterization: {total_circuits} circuits in {n_batches} batches")
    print(f"    Estimated: {char_time_s:.1f}s ({char_time_s/60:.1f} min)")
    print()

    # ── Save results ──
    results = {
        "benchmark": "qpu_timing_v111",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": "Q50",
        "shots": SHOTS,
        "circuit": "EfficientSU2_reps2_4q",
        "batch_10_wall_ms": round(t_batch10 * 1000, 1),
        "batch_1_wall_ms": round(t_batch1 * 1000, 1),
        "per_circuit_ms": round(per_circuit_ms, 1),
        "per_batch_overhead_ms": round(per_batch_overhead_ms, 1),
        "inferred_reset": inferred_reset,
        "projections": {
            "vqe_iter_ms": round(iter_time_ms, 0),
            "vqe_400iter_s": round(campaign_400_s, 1),
            "characterization_12800_s": round(char_time_s, 1),
        }
    }

    out_path = os.path.join(project_dir, "results", "qpu_timing_benchmark.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved: {out_path}")


if __name__ == "__main__":
    main()
