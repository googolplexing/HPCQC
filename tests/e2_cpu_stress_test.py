#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E — E2.1: CPU Parallelism Stress Test.

Tests whether N independent AerSimulator(method="density_matrix")
instances can run concurrently via multiprocessing without result
corruption or crashes. This validates the tiered execution engine's
assumption that small circuits can be parallelized across CPU cores.

Run on LUMI standard partition (CPU only, no GPU):
    srun ... python tests/e2_cpu_stress_test.py

Expected: E2.1 STRESS TEST: ALL CHECKS PASSED

RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0 System 4
"""

import sys
import os
import time
import traceback
import multiprocessing as mp
import numpy as np

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

# Import qiskit at module level so forked children inherit them.
# Without this, each of 128 child processes imports qiskit independently
# from Lustre, causing 10-30s I/O contention per child.
_import_t0 = time.time()
from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
print(f"Qiskit import time: {time.time() - _import_t0:.2f}s (parent only — children inherit via fork)")

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        failed += 1
        errors.append(f"{name}: {detail}")


# ══════════════════════════════════════════════════════════════════════
# Worker function — runs in a separate process
# ══════════════════════════════════════════════════════════════════════

def run_single_vqe(args):
    """Run a minimal density_matrix VQE in an independent process.

    Returns (worker_id, seed, best_energy, num_iters, error_msg).
    """
    worker_id, seed, num_qubits = args
    try:
        # Build a simple TFIM Hamiltonian
        n = num_qubits
        terms = []
        for i in range(n - 1):
            zz = ["I"] * n
            zz[i] = "Z"
            zz[i + 1] = "Z"
            terms.append(("".join(zz), -1.0))
        for i in range(n):
            x = ["I"] * n
            x[i] = "X"
            terms.append(("".join(x), -1.0))
        hamiltonian = SparsePauliOp.from_list(terms)

        # Build ansatz — use function form (Qiskit 2.1+) and decompose
        ansatz = efficient_su2(n, reps=1, entanglement="linear")
        ansatz = ansatz.decompose()
        num_params = ansatz.num_parameters

        # Independent simulator instance — force CPU to avoid ROCm/HSA lock
        sim = AerSimulator(method="density_matrix", device="CPU")

        # Random parameters with deterministic seed
        rng = np.random.default_rng(seed)
        params = rng.uniform(-np.pi, np.pi, num_params)

        # Run 1 energy evaluation (kept minimal for stress test scalability)
        bound = ansatz.assign_parameters(params)
        bound.save_density_matrix()
        result = sim.run(bound, shots=0).result()
        dm = result.data()["density_matrix"]

        # Compute expectation — convert DensityMatrix to numpy array
        h_matrix = hamiltonian.to_matrix()
        dm_array = np.array(dm)
        best_energy = float(np.real(np.trace(h_matrix @ dm_array)))

        return (worker_id, seed, best_energy, 1, None)

    except Exception as e:
        return (worker_id, seed, None, 0, str(e))


# ══════════════════════════════════════════════════════════════════════
print("\n=== E2.1: Sequential Baseline (4 runs) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    t0 = time.time()
    sequential_results = []
    for i in range(4):
        result = run_single_vqe((i, 42 + i, 4))
        sequential_results.append(result)
    t_seq = time.time() - t0

    seq_ok = all(r[4] is None for r in sequential_results)
    check("Sequential: 4 runs complete without error", seq_ok,
          "; ".join(r[4] for r in sequential_results if r[4]))
    check("Sequential: all return valid energies",
          all(r[2] is not None and np.isfinite(r[2]) for r in sequential_results),
          f"energies: {[(r[1], r[2], type(r[2]).__name__) for r in sequential_results]}")
    print(f"    ({t_seq:.2f}s for 4 sequential runs)")
    for r in sequential_results:
        print(f"      seed={r[1]}: energy={r[2]}, error={r[4]}")

    # Store reference energies for reproducibility check
    ref_energies = {r[1]: r[2] for r in sequential_results}

except Exception as e:
    check("E2.1 sequential block", False, f"Exception: {e}")
    traceback.print_exc()
    ref_energies = {}


# ══════════════════════════════════════════════════════════════════════
print("\n=== E2.2: Parallel Execution — 16 Workers ===")
# ══════════════════════════════════════════════════════════════════════
try:
    n_workers = 16
    tasks = [(i, 42 + i, 4) for i in range(n_workers)]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        parallel_results_16 = pool.map(run_single_vqe, tasks)
    t_par_16 = time.time() - t0

    par_ok = all(r[4] is None for r in parallel_results_16)
    check(f"Parallel 16: all {n_workers} runs complete without error",
          par_ok,
          "; ".join(f"w{r[0]}:{r[4]}" for r in parallel_results_16 if r[4]))
    check("Parallel 16: all return valid energies",
          all(r[2] is not None and np.isfinite(r[2]) for r in parallel_results_16))
    print(f"    ({t_par_16:.2f}s for {n_workers} parallel runs)")

    # Reproducibility: same seeds should give same results
    for r in parallel_results_16[:4]:
        seed = r[1]
        if seed in ref_energies and r[2] is not None:
            diff = abs(r[2] - ref_energies[seed])
            check(f"Reproducibility seed {seed}: |ΔE| < 1e-8",
                  diff < 1e-8,
                  f"ΔE = {diff:.2e}")

except Exception as e:
    check("E2.2 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E2.3: Parallel Execution — 64 Workers ===")
# ══════════════════════════════════════════════════════════════════════
try:
    n_workers = 64
    tasks = [(i, 100 + i, 4) for i in range(n_workers)]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        parallel_results_64 = pool.map(run_single_vqe, tasks)
    t_par_64 = time.time() - t0

    par_ok = all(r[4] is None for r in parallel_results_64)
    failed_count = sum(1 for r in parallel_results_64 if r[4] is not None)
    check(f"Parallel 64: all {n_workers} runs complete without error",
          par_ok,
          f"{failed_count} workers failed")
    check("Parallel 64: all return valid energies",
          all(r[2] is not None and np.isfinite(r[2]) for r in parallel_results_64))
    print(f"    ({t_par_64:.2f}s for {n_workers} parallel runs)")

    # Check for result corruption: no two different seeds should give
    # exactly the same energy (probability essentially zero)
    energies = [r[2] for r in parallel_results_64 if r[2] is not None]
    unique_energies = len(set(round(e, 10) for e in energies))
    check("No result corruption (all energies unique)",
          unique_energies == len(energies),
          f"{len(energies)} results but only {unique_energies} unique")

except Exception as e:
    check("E2.3 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E2.4: Parallel Execution — 128 Workers ===")
# ══════════════════════════════════════════════════════════════════════
try:
    n_workers = 128
    tasks = [(i, 200 + i, 4) for i in range(n_workers)]

    t0 = time.time()
    with mp.Pool(n_workers) as pool:
        parallel_results_128 = pool.map(run_single_vqe, tasks)
    t_par_128 = time.time() - t0

    par_ok = all(r[4] is None for r in parallel_results_128)
    failed_count = sum(1 for r in parallel_results_128 if r[4] is not None)
    check(f"Parallel 128: all {n_workers} runs complete without error",
          par_ok,
          f"{failed_count} workers failed")

    success_count = sum(1 for r in parallel_results_128 if r[2] is not None and np.isfinite(r[2]))
    check(f"Parallel 128: {success_count}/{n_workers} return valid energies",
          success_count == n_workers,
          f"only {success_count} valid")

    # Corruption check
    energies = [r[2] for r in parallel_results_128 if r[2] is not None]
    unique_energies = len(set(round(e, 10) for e in energies))
    check("No result corruption at 128 workers",
          unique_energies == len(energies),
          f"{len(energies)} results, {unique_energies} unique")

    print(f"    ({t_par_128:.2f}s for {n_workers} parallel runs)")

    # Scaling report
    if t_seq > 0:
        speedup_16 = (t_seq * 4) / t_par_16 if t_par_16 > 0 else 0
        speedup_64 = (t_seq * 16) / t_par_64 if t_par_64 > 0 else 0
        speedup_128 = (t_seq * 32) / t_par_128 if t_par_128 > 0 else 0
        print(f"\n    Scaling report:")
        print(f"      Sequential (4 runs):  {t_seq:.2f}s")
        print(f"      Parallel 16:          {t_par_16:.2f}s ({speedup_16:.1f}× vs sequential)")
        print(f"      Parallel 64:          {t_par_64:.2f}s ({speedup_64:.1f}× vs sequential)")
        print(f"      Parallel 128:         {t_par_128:.2f}s ({speedup_128:.1f}× vs sequential)")

except Exception as e:
    check("E2.4 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E2.5: Memory Footprint ===")
# ══════════════════════════════════════════════════════════════════════
try:
    import psutil
    proc = psutil.Process(os.getpid())
    mem_mb = proc.memory_info().rss / (1024 * 1024)
    check(f"Parent process memory: {mem_mb:.0f} MB (< 8 GB)",
          mem_mb < 8192,
          f"{mem_mb:.0f} MB")
except ImportError:
    print("    psutil not available — skipping memory check")
except Exception as e:
    check("E2.5 memory check", False, f"Exception: {e}")


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"E2.1 STRESS TEST: {passed} passed, {failed} failed")
if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  ✗ {e}")
    print(f"\nE2.1 STRESS TEST: FAILED ({failed} failures)")
    sys.exit(1)
else:
    print(f"\nE2.1 STRESS TEST: ALL {passed} CHECKS PASSED")
    sys.exit(0)
