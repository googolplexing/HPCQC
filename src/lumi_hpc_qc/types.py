# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Shared data types for the lumi-hpc-qc framework.

Pure dataclasses that flow between modules. No business logic, no external
dependencies beyond stdlib + numpy. Every module in the system can import
from here without creating circular dependencies.
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _generate_experiment_id() -> str:
    """Generate experiment ID: UUID + SLURM job ID (if available).

    Format: {uuid}_{slurm_job_id} or {uuid}_interactive
    The UUID provides global uniqueness. The SLURM job ID provides
    quick cross-reference to SLURM logs and accounting data.
    """
    uid = uuid.uuid4().hex[:12]
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "interactive")
    return f"{uid}_{slurm_job_id}"


# ---------------------------------------------------------------------------
# Configuration types
# ---------------------------------------------------------------------------

@dataclass
class SlurmConfig:
    """SLURM job submission parameters."""
    partition: str = "standard-g"
    account: str = ""
    walltime: str = "01:00:00"
    nodes: int = 1
    gpus_per_node: int = 0  # 0 for CPU-only partitions
    container_path: str = ""
    extra_sbatch_flags: dict[str, str] = field(default_factory=dict)


@dataclass
class CheckpointConfig:
    """Checkpointing configuration."""
    enabled: bool = True
    directory: str = "checkpoints"
    interval: int = 10  # save every N iterations


@dataclass
class ExperimentConfig:
    """Complete experiment configuration — single source of truth.

    Created by config_loader from YAML + CLI overrides. Passed to every
    module that needs to know what to run.
    """
    # Identity
    experiment_id: str = field(default_factory=_generate_experiment_id)

    # Model / physics
    model: str = ""                    # "fermi_hubbard", "heisenberg", etc.
    model_params: dict[str, Any] = field(default_factory=dict)

    # Ansatz
    ansatz: str = ""                   # "hva", "su2", "uccsd", etc.
    ansatz_params: dict[str, Any] = field(default_factory=dict)

    # Optimizer
    optimizer: str = "l_bfgs_b"
    optimizer_params: dict[str, Any] = field(default_factory=dict)

    # Gradient
    gradient: str = "parameter_shift"  # or "finite_difference", "none"
    gradient_params: dict[str, Any] = field(default_factory=dict)

    # Initializer
    initializer: str = "random"        # or "zero", "adiabatic"
    initializer_params: dict[str, Any] = field(default_factory=dict)

    # Backend
    backend: str = "aer_gpu"           # "aer_gpu", "aer_cpu", "iqm_q50", "qcut"
    backend_params: dict[str, Any] = field(default_factory=dict)

    # Error mitigation (None = disabled)
    error_mitigation: str | None = None
    error_mitigation_params: dict[str, Any] = field(default_factory=dict)

    # Simulation
    precision: str = "double"          # "double" (default) or "single"
    num_qubits: int = 0                # computed after Hamiltonian build

    # Execution mode
    mode: str = "interactive"          # "interactive" (Mode A) or "automated" (Mode B)

    # SLURM
    slurm: SlurmConfig = field(default_factory=SlurmConfig)

    # Checkpointing
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)

    # Output
    output_dir: str = "results"


# ---------------------------------------------------------------------------
# Circuit execution types
# ---------------------------------------------------------------------------

@dataclass
class CircuitJob:
    """A batch of circuits to execute on a backend.

    The backend receives this, runs the circuits, and returns CircuitResult.
    """
    job_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    circuits: list[Any] = field(default_factory=list)       # list[QuantumCircuit]
    parameters: list[dict] = field(default_factory=list)    # parameter bindings
    observable: Any = None              # SparsePauliOp or None
    shots: int = 0                      # 0 = statevector (exact)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CircuitResult:
    """Results from executing a CircuitJob."""
    job_id: str = ""
    energies: list[float] | None = None
    counts: list[dict] | None = None
    statevectors: list[Any] | None = None
    execution_time_s: float = 0.0
    backend_name: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Iteration / experiment tracking types
# ---------------------------------------------------------------------------

@dataclass
class IterationRecord:
    """One VQE/VQA optimization iteration."""
    iteration: int = 0
    energy: float = 0.0
    parameters: Any = None              # ndarray, serialized as list for JSON
    gradient_norm: float | None = None
    elapsed_s: float = 0.0
    is_best: bool = False
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ConvergenceSummary:
    """Summary of optimization convergence."""
    total_iterations: int = 0
    best_energy: float = 0.0
    best_iteration: int = 0
    final_energy: float = 0.0
    exact_ground_energy: float | None = None
    absolute_error: float | None = None
    relative_error_pct: float | None = None
    total_circuit_evaluations: int = 0
    total_gradient_evaluations: int = 0
    optimizer_converged: bool = False
    optimizer_message: str = ""


@dataclass
class TimingBreakdown:
    """Phase-level timing with multiple output formats."""
    phases: dict[str, float] = field(default_factory=dict)  # phase → seconds
    total_s: float = 0.0
    percentages: dict[str, float] = field(default_factory=dict)

    def to_human_readable(self) -> str:
        """Pretty-printed table (matches current lumi_vqa format)."""
        lines = [
            "  Phase                                 Duration   % of Total",
            "  ----------------------------------- ---------- ------------",
        ]
        for phase, dur in self.phases.items():
            pct = self.percentages.get(phase, 0.0)
            if dur >= 60:
                dur_str = f"{dur / 60:.1f}m"
            else:
                dur_str = f"{dur:.3f}s"
            lines.append(f"  {phase:<36s} {dur_str:>10s} {pct:>10.1f}%")
        lines.append("  ----------------------------------- ---------- ------------")
        total_str = f"{self.total_s / 60:.1f}m" if self.total_s >= 60 else f"{self.total_s:.3f}s"
        lines.append(f"  {'TOTAL':<36s} {total_str:>10s} {'100.0':>10s}%")
        return "\n".join(lines)

    def to_benchmark_json(self) -> dict[str, Any]:
        """Structured dict for HPC/HPCQC/QC benchmarking."""
        return {
            "phases": self.phases,
            "total_s": self.total_s,
            "percentages": self.percentages,
        }

    def to_training_record(self) -> dict[str, Any]:
        """Flat dict for AI/ML training dataset ingestion.

        Every value is a number or string — no nested structures.
        Designed for direct conversion to a pandas DataFrame row.
        """
        record: dict[str, Any] = {}
        for phase, dur in self.phases.items():
            safe_key = phase.replace(" ", "_").lower()
            record[f"timing_{safe_key}_s"] = dur
            record[f"timing_{safe_key}_pct"] = self.percentages.get(phase, 0.0)
        record["timing_total_s"] = self.total_s
        return record


# ---------------------------------------------------------------------------
# Provenance / reproducibility types
# ---------------------------------------------------------------------------

@dataclass
class ProvenanceData:
    """Reproducibility metadata captured from the runtime environment."""
    # Python + core packages
    python_version: str = ""
    qiskit_version: str = ""
    qiskit_aer_version: str = ""
    numpy_version: str = ""
    scipy_version: str = ""

    # All imported modules (name → version)
    imported_modules: dict[str, str] = field(default_factory=dict)

    # Container
    container_tag: str = ""             # image name/tag, NOT hash (11GB hashing impractical)

    # Hardware
    lumi_node: str = ""                 # hostname
    gpu_model: str | None = None
    gpu_memory_gb: float | None = None
    cpu_model: str = ""
    total_memory_gb: float | None = None

    # SLURM
    slurm_job_id: str = ""
    slurm_partition: str = ""
    slurm_num_nodes: int = 0

    # Code version
    git_commit: str | None = None
    git_branch: str | None = None
    git_dirty: bool = False

    # Quantum hardware (if QPU backend used)
    q50_calibration: dict[str, Any] | None = None

    # Timestamp
    captured_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Circuit metrics (Phase B)
# ---------------------------------------------------------------------------

@dataclass
class CircuitMetrics:
    """Pre- and post-transpilation circuit metrics.

    Captures the impact of topology-aware routing (SWAP insertion)
    on circuit depth and gate count. The difference between pre and post
    metrics quantifies the routing overhead for a given coupling map.
    """
    pre_transpilation_depth: int = 0
    pre_transpilation_gate_count: int = 0
    pre_transpilation_cx_count: int = 0
    post_transpilation_depth: int = 0
    post_transpilation_gate_count: int = 0
    post_transpilation_cx_count: int = 0
    swap_count: int = 0  # post CX - pre CX (approximation)
    coupling_map_source: str = ""
    coupling_map_edges: int = 0
    transpiler_optimization_level: int = 2
    num_parameters: int = 0


# ---------------------------------------------------------------------------
# Full experiment record (final output)
# ---------------------------------------------------------------------------

@dataclass
class ExperimentRecord:
    """Complete experiment output — written to JSON at the end.

    This is the canonical record of what happened. Contains everything
    needed to reproduce the experiment and analyze its results.
    """
    experiment_id: str = ""
    config: ExperimentConfig | None = None
    provenance: ProvenanceData | None = None
    iterations: list[IterationRecord] = field(default_factory=list)
    convergence: ConvergenceSummary | None = None
    timing: TimingBreakdown | None = None
    circuit_metrics: CircuitMetrics | None = None  # Phase B
    noise_config: dict[str, Any] | None = None     # Phase B
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


# ---------------------------------------------------------------------------
# Backend capability declaration
# ---------------------------------------------------------------------------

@dataclass
class BackendCapabilities:
    """What a backend can do — used by controller for routing decisions."""
    max_qubits: int = 0
    supports_statevector: bool = False
    supports_density_matrix: bool = False
    supports_mps: bool = False
    supports_shots: bool = False
    requires_gpu: bool = False
    requires_cpu_only: bool = False
    native_gates: list[str] | None = None
    slurm_partition: str = "standard"   # default to CPU


# ---------------------------------------------------------------------------
# Ansatz metadata (returned by ansatz builders)
# ---------------------------------------------------------------------------

@dataclass
class AnsatzMetadata:
    """Metadata about a built ansatz circuit.

    The workflow reads this to decide gradient strategy, initializer,
    and whether decomposition is needed — no hardcoded if/else chains.
    """
    num_parameters: int = 0
    parameter_names: list[str] = field(default_factory=list)
    gradient_compatibility: str = "parameter_shift"  # or "finite_difference", "both"
    preferred_initializer: str = "random"             # or "zero", "adiabatic"
    requires_decomposition: bool = False
    circuit_depth: int = 0
    gate_counts: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Hamiltonian metadata
# ---------------------------------------------------------------------------

@dataclass
class HamiltonianMetadata:
    """Metadata about a built Hamiltonian."""
    num_qubits: int = 0
    num_pauli_terms: int = 0
    qubit_mapping: str = ""             # "jordan_wigner", "parity", etc.
    description: str = ""               # human-readable summary
    physical_params: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Optimizer result
# ---------------------------------------------------------------------------

@dataclass
class OptimizeResult:
    """Result from an optimizer run."""
    x: Any = None                       # ndarray — optimal parameters
    fun: float = 0.0                    # optimal energy
    nfev: int = 0                       # total function evaluations
    nit: int = 0                        # iterations
    success: bool = False
    message: str = ""
