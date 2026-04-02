#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E — E2: Tiered Execution Planner Validation.

Tests the routing logic (VE10) and CPU parallel dispatch of the
tiered execution planner. Validates that circuits are assigned
to the correct backend based on qubit count and method, and that
CPU batch execution produces correct results.

Run on LUMI standard partition (CPU only):
    srun ... python tests/e2_execution_planner_validation.py

Expected: E2 EXECUTION PLANNER: ALL CHECKS PASSED

RED-SPEC-002 §5 (Tiered Classical Execution Engine)
VE10: 4q→aer_cpu, 12q→aer_gpu routing
"""

import sys
import os
import time

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

import numpy as np

# Import execution planner (no Aer import at module level —
# Aer is only used inside child processes)
from lumi_hpc_qc.sweep.execution_planner import (
    select_backend,
    partition_tasks,
    execute_cpu_batch,
    SimulationTask,
    SimulationResult,
    TieredExecutionPlanner,
)

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
print("\n=== E2.1: VE10 — Backend Routing Logic ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Density matrix routing: ≤8q → CPU, ≥10q → GPU
    check("VE10: 2q density_matrix → aer_cpu",
          select_backend(2, "density_matrix") == "aer_cpu")
    check("VE10: 4q density_matrix → aer_cpu",
          select_backend(4, "density_matrix") == "aer_cpu")
    check("VE10: 6q density_matrix → aer_cpu",
          select_backend(6, "density_matrix") == "aer_cpu")
    check("VE10: 8q density_matrix → aer_cpu",
          select_backend(8, "density_matrix") == "aer_cpu")
    check("VE10: 10q density_matrix → aer_gpu",
          select_backend(10, "density_matrix") == "aer_gpu")
    check("VE10: 12q density_matrix → aer_gpu",
          select_backend(12, "density_matrix") == "aer_gpu")
    check("VE10: 20q density_matrix → aer_gpu",
          select_backend(20, "density_matrix") == "aer_gpu")

    # Statevector routing: ≤18q → CPU, ≥20q → GPU
    check("VE10: 4q statevector → aer_cpu",
          select_backend(4, "statevector") == "aer_cpu")
    check("VE10: 12q statevector → aer_cpu",
          select_backend(12, "statevector") == "aer_cpu")
    check("VE10: 18q statevector → aer_cpu",
          select_backend(18, "statevector") == "aer_cpu")
    check("VE10: 20q statevector → aer_gpu",
          select_backend(20, "statevector") == "aer_gpu")
    check("VE10: 30q statevector → aer_gpu",
          select_backend(30, "statevector") == "aer_gpu")

except Exception as e:
    check("E2.1 block", False, f"Exception: {e}")
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E2.2: Task Partitioning ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Create a mixed batch of tasks
    tasks = [
        SimulationTask(task_id="t_2q",  num_qubits=2,  method="density_matrix"),
        SimulationTask(task_id="t_4q",  num_qubits=4,  method="density_matrix"),
        SimulationTask(task_id="t_8q",  num_qubits=8,  method="density_matrix"),
        SimulationTask(task_id="t_12q", num_qubits=12, method="density_matrix"),
        SimulationTask(task_id="t_20q", num_qubits=20, method="density_matrix"),
        SimulationTask(task_id="t_sv4", num_qubits=4,  method="statevector"),
        SimulationTask(task_id="t_sv20", num_qubits=20, method="statevector"),
    ]

    cpu_tasks, gpu_tasks = partition_tasks(tasks)

    check("Partition: 4 CPU tasks (2q, 4q, 8q dm + 4q sv)",
          len(cpu_tasks) == 4,
          f"got {len(cpu_tasks)}")
    check("Partition: 3 GPU tasks (12q, 20q dm + 20q sv)",
          len(gpu_tasks) == 3,
          f"got {len(gpu_tasks)}")

    # Verify assigned_backend set correctly
    check("Partition: all CPU tasks have assigned_backend='aer_cpu'",
          all(t.assigned_backend == "aer_cpu" for t in cpu_tasks))
    check("Partition: all GPU tasks have assigned_backend='aer_gpu'",
          all(t.assigned_backend == "aer_gpu" for t in gpu_tasks))

    # Verify specific tasks went to correct backend
    task_map = {t.task_id: t for t in cpu_tasks + gpu_tasks}
    check("Partition: t_4q → aer_cpu",
          task_map["t_4q"].assigned_backend == "aer_cpu")
    check("Partition: t_12q → aer_gpu",
          task_map["t_12q"].assigned_backend == "aer_gpu")

except Exception as e:
    check("E2.2 block", False, f"Exception: {e}")
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E2.3: Routing Summary ===")
# ══════════════════════════════════════════════════════════════════════
try:
    planner = TieredExecutionPlanner(cpu_workers=16)
    summary = planner.summary(tasks)

    check("Summary: total_tasks = 7",
          summary["total_tasks"] == 7)
    check("Summary: cpu_tasks = 4",
          summary["cpu_tasks"] == 4,
          f"got {summary['cpu_tasks']}")
    check("Summary: gpu_tasks = 3",
          summary["gpu_tasks"] == 3,
          f"got {summary['gpu_tasks']}")
    check("Summary: cpu_qubit_range = (2, 8)",
          summary["cpu_qubit_range"] == (2, 8),
          f"got {summary['cpu_qubit_range']}")
    check("Summary: gpu_qubit_range = (12, 20)",
          summary["gpu_qubit_range"] == (12, 20),
          f"got {summary['gpu_qubit_range']}")

except Exception as e:
    check("E2.3 block", False, f"Exception: {e}")
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E2.4: CPU Parallel Execution (4q TFIM, 16 tasks) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Build a simple TFIM 4q circuit + Hamiltonian for CPU execution.
    # Imports here are inherited by forked children via module-level.
    from qiskit.circuit.library import efficient_su2
    from qiskit.quantum_info import SparsePauliOp
    print("    Qiskit imported for task preparation (children inherit via fork)")

    n = 4
    terms = []
    for i in range(n - 1):
        zz = ["I"] * n
        zz[i] = "Z"; zz[i + 1] = "Z"
        terms.append(("".join(zz), -1.0))
    for i in range(n):
        x = ["I"] * n
        x[i] = "X"
        terms.append(("".join(x), -1.0))
    hamiltonian = SparsePauliOp.from_list(terms)

    ansatz = efficient_su2(n, reps=1, entanglement="linear").decompose()
    num_params = ansatz.num_parameters

    # Create 16 tasks with different seeds
    cpu_sim_tasks = []
    for i in range(16):
        rng = np.random.default_rng(42 + i)
        params = rng.uniform(-np.pi, np.pi, num_params)
        cpu_sim_tasks.append(SimulationTask(
            task_id=f"cpu_{i:03d}",
            num_qubits=n,
            method="density_matrix",
            shots=0,
            circuit=ansatz,
            parameters=params,
            observable=hamiltonian,
            seed=42 + i,
        ))

    # Execute in parallel
    t0 = time.time()
    results = execute_cpu_batch(cpu_sim_tasks, workers=16)
    t_par = time.time() - t0

    check("CPU batch: 16 results returned",
          len(results) == 16,
          f"got {len(results)}")
    check("CPU batch: no errors",
          all(r.error is None for r in results),
          "; ".join(f"{r.task_id}:{r.error}" for r in results if r.error))
    check("CPU batch: all energies finite",
          all(r.energy is not None and np.isfinite(r.energy) for r in results),
          f"energies: {[r.energy for r in results]}")
    check("CPU batch: all energies unique (different seeds)",
          len(set(round(r.energy, 10) for r in results if r.energy)) == 16,
          "duplicate energies detected")
    check("CPU batch: all backend_used = 'aer_cpu'",
          all(r.backend_used == "aer_cpu" for r in results))
    print(f"    ({t_par:.2f}s for 16 parallel tasks)")

    # Reproducibility: same seed should give same energy
    results2 = execute_cpu_batch(cpu_sim_tasks[:4], workers=4)
    for r1, r2 in zip(results[:4], results2):
        diff = abs(r1.energy - r2.energy)
        check(f"Reproducibility {r1.task_id}: |ΔE| < 1e-8",
              diff < 1e-8,
              f"ΔE = {diff:.2e}")

except Exception as e:
    check("E2.4 block", False, f"Exception: {e}")
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E2.5: Planner End-to-End (CPU-only mode) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Mixed batch: some CPU, some GPU. GPU tasks get skipped (no GPU on standard).
    mixed_tasks = []

    # 4 CPU tasks (4q)
    for i in range(4):
        rng = np.random.default_rng(100 + i)
        params = rng.uniform(-np.pi, np.pi, num_params)
        mixed_tasks.append(SimulationTask(
            task_id=f"mix_cpu_{i}",
            num_qubits=4,
            method="density_matrix",
            shots=0,
            circuit=ansatz,
            parameters=params,
            observable=hamiltonian,
            seed=100 + i,
        ))

    # 2 GPU tasks (12q) — will be skipped in CPU-only mode
    mixed_tasks.append(SimulationTask(
        task_id="mix_gpu_0", num_qubits=12, method="density_matrix"))
    mixed_tasks.append(SimulationTask(
        task_id="mix_gpu_1", num_qubits=12, method="density_matrix"))

    planner = TieredExecutionPlanner(cpu_workers=4)
    results = planner.execute(mixed_tasks, gpu_executor=None)

    check("E2E: 6 results returned (4 CPU + 2 GPU skipped)",
          len(results) == 6)
    check("E2E: CPU tasks have valid energies",
          all(r.energy is not None for r in results[:4]))
    check("E2E: GPU tasks have error (no executor)",
          all(r.error is not None for r in results[4:]),
          f"GPU results: {[(r.task_id, r.error) for r in results[4:]]}")
    check("E2E: result order matches input task order",
          [r.task_id for r in results] == [t.task_id for t in mixed_tasks])

except Exception as e:
    check("E2.5 block", False, f"Exception: {e}")
    import traceback; traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"E2 EXECUTION PLANNER: {passed} passed, {failed} failed")
if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  ✗ {e}")
    print(f"\nE2 EXECUTION PLANNER: FAILED ({failed} failures)")
    sys.exit(1)
else:
    print(f"\nE2 EXECUTION PLANNER: ALL {passed} CHECKS PASSED")
    sys.exit(0)
