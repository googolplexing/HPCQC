# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Tiered execution planner — routes simulations to CPU or GPU.

Distributes Aer simulation tasks across LUMI's CPU (standard) and GPU
(standard-g) partitions based on circuit size and simulation method.
Small circuits are faster on CPU due to zero GPU kernel launch overhead.
Large circuits need GPU memory and parallelism.

RED-SPEC-002 §5 (Tiered Classical Execution Engine)
RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0 System 4

Routing thresholds (from benchmarking and hardware characteristics):

    Method            | CPU (standard)  | GPU (standard-g)
    ------------------|-----------------|------------------
    density_matrix    | 2q–8q           | 10q+
    statevector       | 2q–18q          | 20q+

CPU execution uses multiprocessing.Pool for parallelism (128 workers
on a LUMI-C node = 1 worker per physical core).

CRITICAL CONSTRAINT: Aer's C++ thread pool must never be initialized
in the parent process before forking. All Aer execution happens in
child processes. See debug_e2_fork_order.py for the proof.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable


# ═══════════════════════════════════════════════════════════════════════
# Routing thresholds
# ═══════════════════════════════════════════════════════════════════════

# Density matrix: 2^n × 2^n × 16 bytes (complex128)
# At 8q: 256 × 256 × 16 = 1 MB (CPU fast)
# At 10q: 1024 × 1024 × 16 = 16 MB (GPU break-even)
_DM_CPU_MAX_QUBITS = 8

# Statevector: 2^n × 16 bytes (complex128)
# At 18q: 262,144 × 16 = 4 MB (CPU fast)
# At 20q: 1,048,576 × 16 = 16 MB (GPU break-even)
_SV_CPU_MAX_QUBITS = 18

# Default parallel workers (matches LUMI-C full node: 2 × 64 cores)
_DEFAULT_CPU_WORKERS = 128


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SimulationTask:
    """A single simulation to execute in the sweep.

    Carries all context needed for one (placement × environment × seed)
    simulation. The execution planner routes this to the appropriate backend.
    """
    task_id: str = ""
    num_qubits: int = 0
    method: str = "density_matrix"    # "density_matrix" or "statevector"
    shots: int = 0                    # 0 = exact (statevector/dm expectation)

    # Placement context
    placement_indices: list[int] = field(default_factory=list)
    device_prefix: str = ""
    topology_hash: str = ""

    # Noise / calibration context
    noise_environment: str = ""       # e.g., "noiseless", "noise_full"
    calibration_id: str = ""

    # Circuit (set by sweep engine before dispatch)
    circuit: Any = None               # QuantumCircuit (bound or unbound)
    parameters: Any = None            # ndarray of parameter values
    observable: Any = None            # SparsePauliOp for expectation value

    # Noise context (v1.1.1 — Level 1 E2/E7 integration)
    noise_model: Any = None           # qiskit_aer.noise.NoiseModel or None
    coupling_map: Any = None          # qiskit CouplingMap or None

    # Execution metadata (filled by planner after routing)
    assigned_backend: str = ""
    seed: int = 0


@dataclass
class SimulationResult:
    """Result of executing a SimulationTask."""
    task_id: str = ""
    energy: float | None = None
    counts: dict[str, int] | None = None
    execution_time_s: float = 0.0
    backend_used: str = ""
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# Routing
# ═══════════════════════════════════════════════════════════════════════

def select_backend(num_qubits: int, method: str = "density_matrix") -> str:
    """Select the optimal backend for a given circuit size and method.

    RED-SPEC-002 §5.1

    Args:
        num_qubits: Circuit qubit count.
        method: Simulation method — "density_matrix" or "statevector".

    Returns:
        Backend name: "aer_cpu" or "aer_gpu".
    """
    if method == "statevector":
        return "aer_cpu" if num_qubits <= _SV_CPU_MAX_QUBITS else "aer_gpu"
    else:  # density_matrix
        return "aer_cpu" if num_qubits <= _DM_CPU_MAX_QUBITS else "aer_gpu"


def partition_tasks(
    tasks: list[SimulationTask],
) -> tuple[list[SimulationTask], list[SimulationTask]]:
    """Partition tasks into CPU and GPU batches.

    Returns:
        (cpu_tasks, gpu_tasks) with assigned_backend set on each task.
    """
    cpu_tasks = []
    gpu_tasks = []
    for task in tasks:
        backend = select_backend(task.num_qubits, task.method)
        task.assigned_backend = backend
        if backend == "aer_cpu":
            cpu_tasks.append(task)
        else:
            gpu_tasks.append(task)
    return cpu_tasks, gpu_tasks


# ═══════════════════════════════════════════════════════════════════════
# CPU parallel execution
# ═══════════════════════════════════════════════════════════════════════

def _cpu_worker(args: tuple) -> dict:
    """Execute a single simulation task in a child process.

    This function runs in a forked child. Aer is imported at module level
    (parent inherits via fork COW). The child creates its own AerSimulator
    instance — no shared state with other workers.

    v1.1.1: Accepts optional noise_model and coupling_map for noisy
    simulation (Level 1 E2/E7 integration).

    Returns a dict (not SimulationResult) for pickle compatibility.
    """
    import numpy as np
    from qiskit_aer import AerSimulator

    (task_id, circuit, parameters, observable, num_qubits,
     method, shots, seed, noise_model, coupling_map) = args

    try:
        t0 = time.time()

        # Build simulator with optional noise model
        sim_kwargs: dict[str, Any] = {"method": method, "device": "CPU"}
        if noise_model is not None:
            sim_kwargs["noise_model"] = noise_model
        if coupling_map is not None:
            sim_kwargs["coupling_map"] = coupling_map

        sim = AerSimulator(**sim_kwargs)

        # Bind parameters if needed
        if parameters is not None and circuit.num_parameters > 0:
            bound = circuit.assign_parameters(parameters)
        else:
            bound = circuit

        # Attach observable for expectation value
        if observable is not None and shots == 0:
            bound.save_expectation_value(
                observable,
                list(range(num_qubits)),
                label="energy",
            )

        result = sim.run(bound, shots=shots, seed_simulator=seed).result()

        energy = None
        counts = None
        if shots == 0 and observable is not None:
            energy = float(np.real(result.data()["energy"]))
        elif shots > 0:
            counts = result.get_counts()

        elapsed = time.time() - t0
        return {
            "task_id": task_id,
            "energy": energy,
            "counts": counts,
            "execution_time_s": elapsed,
            "backend_used": "aer_cpu",
            "error": None,
        }

    except Exception as e:
        return {
            "task_id": task_id,
            "energy": None,
            "counts": None,
            "execution_time_s": 0.0,
            "backend_used": "aer_cpu",
            "error": str(e),
        }


def execute_cpu_batch(
    tasks: list[SimulationTask],
    workers: int = _DEFAULT_CPU_WORKERS,
) -> list[SimulationResult]:
    """Execute CPU tasks in parallel via multiprocessing.Pool.

    CRITICAL: This function must not be called after any Aer .run()
    in the parent process. Aer's C++ thread pool causes fork deadlock.
    See debug_e2_fork_order.py for proof.

    Args:
        tasks: List of SimulationTask with circuit, parameters, observable set.
        workers: Number of parallel workers (default 128 = LUMI-C full node).

    Returns:
        List of SimulationResult in same order as input tasks.
    """
    if not tasks:
        return []

    # Prepare picklable args (dataclasses with QuantumCircuit don't pickle cleanly)
    worker_args = []
    for task in tasks:
        worker_args.append((
            task.task_id,
            task.circuit,
            task.parameters,
            task.observable,
            task.num_qubits,
            task.method,
            task.shots,
            task.seed,
            task.noise_model,
            task.coupling_map,
        ))

    # Clamp workers to task count — no point having idle workers
    actual_workers = min(workers, len(worker_args))

    t0 = time.time()
    with mp.Pool(actual_workers) as pool:
        raw_results = pool.map(_cpu_worker, worker_args)
    total_time = time.time() - t0

    # Convert dicts back to SimulationResult
    results = []
    for r in raw_results:
        results.append(SimulationResult(
            task_id=r["task_id"],
            energy=r["energy"],
            counts=r["counts"],
            execution_time_s=r["execution_time_s"],
            backend_used=r["backend_used"],
            error=r["error"],
            metadata={"parallel_total_time_s": total_time},
        ))

    return results


# ═══════════════════════════════════════════════════════════════════════
# Execution planner (top-level orchestrator)
# ═══════════════════════════════════════════════════════════════════════

class TieredExecutionPlanner:
    """Routes and executes simulation tasks across CPU and GPU backends.

    Usage:
        planner = TieredExecutionPlanner(cpu_workers=128)
        results = planner.execute(tasks)

    The planner:
    1. Partitions tasks into CPU and GPU batches (select_backend)
    2. Executes CPU batch in parallel via multiprocessing.Pool
    3. Executes GPU batch via aer_gpu (sequential, GPU does parallelism)
    4. Merges results and returns them in original task order
    """

    def __init__(
        self,
        cpu_workers: int = _DEFAULT_CPU_WORKERS,
        gpu_backend_name: str = "aer_gpu",
    ) -> None:
        self.cpu_workers = cpu_workers
        self.gpu_backend_name = gpu_backend_name

    def route(self, tasks: list[SimulationTask]) -> dict[str, list[SimulationTask]]:
        """Partition tasks and return routing plan without executing.

        Returns:
            {"aer_cpu": [...], "aer_gpu": [...]} with assigned_backend set.
        """
        cpu_tasks, gpu_tasks = partition_tasks(tasks)
        return {"aer_cpu": cpu_tasks, "aer_gpu": gpu_tasks}

    def execute(
        self,
        tasks: list[SimulationTask],
        gpu_executor: Callable[[list[SimulationTask]], list[SimulationResult]] | None = None,
    ) -> list[SimulationResult]:
        """Route and execute all tasks. Returns results in original order.

        Args:
            tasks: Simulation tasks to execute.
            gpu_executor: Optional callable for GPU tasks. If None, GPU tasks
                         are skipped (useful for CPU-only testing).

        Returns:
            Results in same order as input tasks.
        """
        # Build task index for reordering
        task_order = {task.task_id: i for i, task in enumerate(tasks)}

        # Partition
        routing = self.route(tasks)
        cpu_tasks = routing["aer_cpu"]
        gpu_tasks = routing["aer_gpu"]

        print(f"  Execution plan: {len(cpu_tasks)} CPU + {len(gpu_tasks)} GPU tasks")
        if cpu_tasks:
            qubits_cpu = sorted(set(t.num_qubits for t in cpu_tasks))
            print(f"    CPU ({self.cpu_workers} workers): "
                  f"{len(cpu_tasks)} tasks, qubits: {qubits_cpu}")
        if gpu_tasks:
            qubits_gpu = sorted(set(t.num_qubits for t in gpu_tasks))
            print(f"    GPU: {len(gpu_tasks)} tasks, qubits: {qubits_gpu}")

        # Execute CPU batch
        cpu_results = execute_cpu_batch(cpu_tasks, self.cpu_workers) if cpu_tasks else []

        # Execute GPU batch
        if gpu_tasks and gpu_executor is not None:
            gpu_results = gpu_executor(gpu_tasks)
        elif gpu_tasks:
            print(f"    WARNING: {len(gpu_tasks)} GPU tasks skipped (no gpu_executor)")
            gpu_results = [
                SimulationResult(
                    task_id=t.task_id,
                    error="GPU executor not provided",
                    backend_used="aer_gpu",
                )
                for t in gpu_tasks
            ]
        else:
            gpu_results = []

        # Merge and reorder to match input
        all_results = cpu_results + gpu_results
        result_map = {r.task_id: r for r in all_results}
        ordered = [result_map[task.task_id] for task in tasks]

        return ordered

    def summary(self, tasks: list[SimulationTask]) -> dict[str, Any]:
        """Generate a routing summary without executing.

        Returns:
            Dict with task counts, qubit distributions, backend assignments.
        """
        routing = self.route(tasks)
        cpu_tasks = routing["aer_cpu"]
        gpu_tasks = routing["aer_gpu"]

        return {
            "total_tasks": len(tasks),
            "cpu_tasks": len(cpu_tasks),
            "gpu_tasks": len(gpu_tasks),
            "cpu_qubit_range": (
                min(t.num_qubits for t in cpu_tasks) if cpu_tasks else None,
                max(t.num_qubits for t in cpu_tasks) if cpu_tasks else None,
            ),
            "gpu_qubit_range": (
                min(t.num_qubits for t in gpu_tasks) if gpu_tasks else None,
                max(t.num_qubits for t in gpu_tasks) if gpu_tasks else None,
            ),
            "cpu_methods": sorted(set(t.method for t in cpu_tasks)),
            "gpu_methods": sorted(set(t.method for t in gpu_tasks)),
        }
