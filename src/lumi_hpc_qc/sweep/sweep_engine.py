# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""E7 sweep engine orchestrator — YAML config to HDF5 noise atlas.

Connects all Phase E subsystems into a single pipeline:

    YAML config
      → Grid expansion (hamiltonians × topologies × seeds × ...)
      → E1: placement solver (find all valid placements per device)
      → E4: twin simulator (11 envs per placement per calibration)
      → E2: tiered execution (route CPU/GPU, parallel dispatch)
      → E3: HDF5 writer (WAL-safe, write-during-execution)
      → E6a: packing (multi-round QPU batching, when available)

Researcher-facing interface: a single YAML file defines the entire sweep.
The engine expands this into potentially hundreds of thousands of tasks,
dispatches them across LUMI's CPU and GPU partitions, and writes a single
HDF5 file containing the complete noise atlas.

RED-SPEC-002 §§1–17
RED-DIRECTIVE-E4-SCHEMA-v1.0 (61-column Parquet schema)
BLUE-RESP-PHASE-E-ROADMAP-v1.0 §8.2

Validation targets:
  VE18: Full sweep from YAML → HDF5 (TFIM 4q, 1 cal, all placements, all envs, 2 seeds)
  VE19: Tiered measurement stats intervals respected in sweep output
  VE22: Topology library integrated — chain + star placements in same sweep
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lumi_hpc_qc.sweep.noise_configs import (
    NOISE_ENVIRONMENTS,
    NOISE_ENV_BY_NAME,
    NoiseConfig,
)
from lumi_hpc_qc.sweep.placement_solver import (
    GeneralPlacementSolver,
    Placement,
    PackingRound,
)
from lumi_hpc_qc.sweep.topology_library import (
    TOPOLOGY_LIBRARY,
    get_topologies_for_size,
    list_all_topologies,
)
from lumi_hpc_qc.sweep.twin_simulator import (
    run_twin_battery,
    run_multi_calibration_battery,
    TwinResult,
    PlacementBatteryResult,
)
from lumi_hpc_qc.data.hdf5_writer import SweepHDF5Writer, SweepResultEntry
from lumi_hpc_qc import __version__ as _pkg_version


# ═══════════════════════════════════════════════════════════════════════
# Configuration data types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SweepExperimentConfig:
    """One experiment block from the sweep YAML.

    Expanded from the YAML 'experiments' list. Each block specifies
    a set of hamiltonians, topologies, seeds, and noise configs to
    sweep across all valid placements.
    """
    experiment_type: str = "characterization"  # "characterization" or "vqe_sweep"

    # What to sweep
    hamiltonians: list[str] = field(default_factory=list)
    qubit_sizes: list[int] = field(default_factory=list)
    topologies: str | list[str] = "auto"  # "auto" or explicit list
    seeds: int = 20
    seed_offset: int = 0  # Starting seed (seeds run from offset to offset+seeds-1)

    # Noise scope
    noise_configs: str | list[str] = "all"  # "all" or explicit list
    measurement_stats_interval_override: int | None = None  # Override per-env default

    # Placement strategy
    placement: str | int = "all_valid"  # "all_valid", "top_N", or int

    # VQE-specific grid (only for vqe_sweep type)
    grid: dict[str, list[Any]] = field(default_factory=dict)

    # Metadata
    label: str = ""


@dataclass
class SweepConfig:
    """Complete sweep configuration parsed from YAML."""
    experiments: list[SweepExperimentConfig] = field(default_factory=list)
    calibrations: list[str] = field(default_factory=list)

    # Execution limits
    cpu_workers: int = 128
    gpu_workers: int = 8

    # Output
    output_dir: str = "sweep_output"
    hdf5_filename: str = "sweep.h5"
    enable_swmr: bool = False
    debug_json: bool = False

    # Sweep metadata
    sweep_id: str = ""
    framework_version: str = _pkg_version


# ═══════════════════════════════════════════════════════════════════════
# Grid expansion
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SweepTask:
    """One atomic unit of work in the sweep.

    Represents a single (hamiltonian, qubit_size, topology, seed,
    calibration, placement) combination that needs to run through
    the full 11-environment twin battery.
    """
    task_id: str = ""
    hamiltonian: str = ""
    qubit_size: int = 0
    topology_name: str = ""
    topology_edges: list[tuple[int, int]] = field(default_factory=list)
    seed: int = 0
    calibration_path: str = ""
    calibration_id: str = ""
    experiment_type: str = "characterization"
    noise_configs: list[NoiseConfig] = field(default_factory=list)
    placement_strategy: str = "all_valid"
    max_placements: int | None = None
    label: str = ""


def expand_grid(config: SweepConfig) -> list[SweepTask]:
    """Expand sweep config into individual tasks.

    Each SweepTask represents one (hamiltonian × topology × seed ×
    calibration) combination. The twin battery (11 envs × all placements)
    runs within each task — placements are found at execution time.

    Returns:
        List of SweepTask, ordered for optimal cache locality:
        group by (hamiltonian, topology) first so placement solver
        results are reused across seeds and calibrations.
    """
    tasks: list[SweepTask] = []
    task_counter = 0

    for exp in config.experiments:
        # Resolve noise configs
        if exp.noise_configs == "all" or exp.noise_configs == ["all"]:
            noise_envs = list(NOISE_ENVIRONMENTS)
        else:
            nc_names = exp.noise_configs if isinstance(exp.noise_configs, list) else [exp.noise_configs]
            noise_envs = [NOISE_ENV_BY_NAME[n] for n in nc_names if n in NOISE_ENV_BY_NAME]

        # Apply measurement_stats_interval override from YAML (Item 4)
        if exp.measurement_stats_interval_override is not None:
            from dataclasses import replace
            noise_envs = [
                replace(env, measurement_stats_interval=exp.measurement_stats_interval_override)
                for env in noise_envs
            ]

        # Resolve placement strategy
        placement_strategy = "all_valid"
        max_placements = None
        if isinstance(exp.placement, int):
            placement_strategy = "top_n"
            max_placements = exp.placement
        elif isinstance(exp.placement, str) and exp.placement.startswith("top_"):
            placement_strategy = "top_n"
            max_placements = int(exp.placement.split("_")[1])
        else:
            placement_strategy = str(exp.placement)

        # Resolve topologies for each qubit size
        for qsize in exp.qubit_sizes:
            if exp.topologies == "auto":
                topos = get_topologies_for_size(qsize)
                if not topos:
                    # Fallback: linear chain
                    topo_name = f"{qsize}q_chain"
                    topos = {topo_name: {
                        "qubits": qsize,
                        "edges": [(i, i + 1) for i in range(qsize - 1)],
                    }}
            elif isinstance(exp.topologies, list):
                topos = {
                    name: TOPOLOGY_LIBRARY[name]
                    for name in exp.topologies
                    if name in TOPOLOGY_LIBRARY
                }
            else:
                topos = get_topologies_for_size(qsize)

            for ham in exp.hamiltonians:
                for topo_name, topo_spec in topos.items():
                    for cal_path in config.calibrations:
                        for seed_idx in range(exp.seeds):
                            task_counter += 1
                            tasks.append(SweepTask(
                                task_id=f"T{task_counter:06d}",
                                hamiltonian=ham,
                                qubit_size=qsize,
                                topology_name=topo_name,
                                topology_edges=list(topo_spec["edges"]),
                                seed=exp.seed_offset + seed_idx,
                                calibration_path=cal_path,
                                calibration_id=_calibration_id(cal_path),
                                experiment_type=exp.experiment_type,
                                noise_configs=noise_envs,
                                placement_strategy=placement_strategy,
                                max_placements=max_placements,
                                label=exp.label,
                            ))

    return tasks


def _calibration_id(path: str) -> str:
    """Extract a short calibration ID from a file path."""
    name = Path(path).stem
    # e.g., "q50_calibration_20260330" → "cal_20260330"
    if "_" in name:
        parts = name.split("_")
        # Take last part that looks like a date
        for p in reversed(parts):
            if p.isdigit() and len(p) == 8:
                return f"cal_{p}"
    return f"cal_{hashlib.md5(path.encode()).hexdigest()[:8]}"


# ═══════════════════════════════════════════════════════════════════════
# Config parsing
# ═══════════════════════════════════════════════════════════════════════

def parse_sweep_config(yaml_dict: dict[str, Any]) -> SweepConfig:
    """Parse a YAML dict into a SweepConfig.

    Accepts the 'sweep' section of the config file. If the top-level
    dict has a 'sweep' key, extracts it. Otherwise treats the whole
    dict as the sweep config.

    Args:
        yaml_dict: Parsed YAML dictionary.

    Returns:
        Validated SweepConfig.

    Raises:
        ValueError: If required fields are missing or invalid.
    """
    # Extract sweep section if present
    sweep_dict = yaml_dict.get("sweep", yaml_dict)

    # Parse experiments
    experiments = []
    for exp_dict in sweep_dict.get("experiments", []):
        exp = SweepExperimentConfig(
            experiment_type=exp_dict.get("type", "characterization"),
            hamiltonians=exp_dict.get("hamiltonians", []),
            qubit_sizes=exp_dict.get("qubit_sizes", []),
            topologies=exp_dict.get("topologies", "auto"),
            seeds=exp_dict.get("seeds", 20),
            seed_offset=exp_dict.get("seed_offset", 0),
            noise_configs=exp_dict.get("noise_configs", "all"),
            measurement_stats_interval_override=exp_dict.get(
                "measurement_stats_interval", None
            ),
            placement=exp_dict.get("placement", "all_valid"),
            grid=exp_dict.get("grid", {}),
            label=exp_dict.get("label", ""),
        )
        experiments.append(exp)

    # Parse execution section
    execution = sweep_dict.get("execution", {})

    config = SweepConfig(
        experiments=experiments,
        calibrations=sweep_dict.get("calibrations", []),
        cpu_workers=execution.get("cpu_workers", 128),
        gpu_workers=execution.get("gpu_workers", 8),
        output_dir=sweep_dict.get("output_dir", "sweep_output"),
        hdf5_filename=sweep_dict.get("hdf5_filename", "sweep.h5"),
        enable_swmr=sweep_dict.get("enable_swmr", False),
        debug_json=sweep_dict.get("debug_json", False),
        sweep_id=sweep_dict.get("sweep_id", str(uuid.uuid4())[:8]),
        framework_version=sweep_dict.get("framework_version", _pkg_version),
    )

    return config


def validate_sweep_config(config: SweepConfig) -> list[str]:
    """Validate a SweepConfig for completeness and consistency.

    Returns:
        List of error messages. Empty = valid.
    """
    errors: list[str] = []

    if not config.experiments:
        errors.append("No experiments defined in sweep config")

    if not config.calibrations:
        errors.append("No calibration files specified")

    for i, exp in enumerate(config.experiments):
        prefix = f"experiment[{i}]"
        if not exp.hamiltonians:
            errors.append(f"{prefix}: no hamiltonians specified")
        if not exp.qubit_sizes:
            errors.append(f"{prefix}: no qubit_sizes specified")
        if exp.seeds < 1:
            errors.append(f"{prefix}: seeds must be >= 1, got {exp.seeds}")

        # Validate noise config names
        if isinstance(exp.noise_configs, list) and exp.noise_configs != ["all"]:
            for nc_name in exp.noise_configs:
                if nc_name not in NOISE_ENV_BY_NAME:
                    errors.append(
                        f"{prefix}: unknown noise config '{nc_name}'. "
                        f"Available: {', '.join(NOISE_ENV_BY_NAME.keys())}"
                    )

        # Validate topology names if explicit
        if isinstance(exp.topologies, list):
            for topo_name in exp.topologies:
                if topo_name not in TOPOLOGY_LIBRARY:
                    errors.append(
                        f"{prefix}: unknown topology '{topo_name}'. "
                        f"Available: {', '.join(list_all_topologies())}"
                    )

    # Validate calibration files exist
    for cal_path in config.calibrations:
        if not Path(cal_path).exists():
            errors.append(f"Calibration file not found: {cal_path}")

    return errors


# ═══════════════════════════════════════════════════════════════════════
# Sweep execution engine
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class SweepProgress:
    """Live progress tracking for the sweep."""
    total_tasks: int = 0
    completed_tasks: int = 0
    total_placements: int = 0
    total_simulations: int = 0
    total_deduplicated: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0
    hdf5_writes: int = 0


@dataclass
class SweepResult:
    """Final result of a complete sweep run."""
    sweep_id: str = ""
    hdf5_path: str = ""
    total_tasks: int = 0
    total_placements: int = 0
    total_simulations: int = 0
    total_deduplicated: int = 0
    total_hdf5_writes: int = 0
    total_errors: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)
    placement_summary: dict[str, int] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════
# Parallel battery worker (E2/E7 Level 1 integration)
# ═══════════════════════════════════════════════════════════════════════

def _battery_worker(args: tuple) -> dict[str, Any]:
    """Run one twin battery in a worker process.

    Module-level function for multiprocessing.Pool (must be picklable).
    Each worker runs all noise environments for one (seed, placement).

    Args:
        Tuple of: (circuit, observable, qubit_names, cal_json, cal_id,
                   placement_id_str, topology_hash, noise_envs, seed,
                   device, noiseless_cache)

    Returns:
        Dict with battery results and metadata for HDF5 writing.
    """
    (circuit, observable, qubit_names, cal_json, cal_id,
     placement_id_str, topology_hash, noise_envs, seed,
     device, noiseless_cache) = args

    try:
        battery = run_twin_battery(
            circuit=circuit,
            observable=observable,
            qubit_names=qubit_names,
            calibration_data=cal_json,
            calibration_id=cal_id,
            placement_id=placement_id_str,
            topology_hash=topology_hash,
            environments=noise_envs,
            seed=seed,
            device=device,
            noiseless_cache=noiseless_cache,
        )
        return {
            "placement_id_str": placement_id_str,
            "battery": battery,
            "error": None,
        }
    except Exception as e:
        return {
            "placement_id_str": placement_id_str,
            "battery": None,
            "error": str(e),
        }


class SweepEngine:
    """Top-level sweep orchestrator.

    Connects E1 (placement), E4 (twin sim), E2 (tiered exec),
    E3 (HDF5 writer), and E6a (packing) into a single pipeline.

    Usage:
        engine = SweepEngine(config)
        result = engine.run()
    """

    def __init__(
        self,
        config: SweepConfig,
        *,
        device: str = "CPU",
        progress_callback: Any | None = None,
    ) -> None:
        self._config = config
        self._device = device
        self._progress_callback = progress_callback
        self._progress = SweepProgress()

        # E1: Placement solver (shared across tasks with same device)
        self._solver = GeneralPlacementSolver()

        # E2: Tiered execution planner (v1.1.1 — Level 1 integration)
        from lumi_hpc_qc.sweep.execution_planner import TieredExecutionPlanner
        self._planner = TieredExecutionPlanner(
            cpu_workers=config.cpu_workers,
        )

        # E4: Noiseless deduplication cache (shared across calibrations)
        self._noiseless_cache: dict[str, TwinResult] = {}

        # Calibration data cache: cal_path → (cal_id, cal_json_dict, DeviceCalibration)
        self._cal_cache: dict[str, tuple[str, dict[str, Any], Any]] = {}

        # Placement cache: (topology_name, device_id) → list[Placement]
        self._placement_cache: dict[tuple[str, str], list[Placement]] = {}

    def run(self) -> SweepResult:
        """Execute the complete sweep. Returns SweepResult.

        Steps:
          1. Parse and validate config
          2. Load calibrations, register devices
          3. Expand grid into tasks
          4. Group tasks by (hamiltonian, topology, calibration) for cache locality
          5. For each group:
             a. Build circuit + observable
             b. Find placements (E1, cached)
             c. For each seed:
                - For each placement:
                  - Run twin battery (E4, with noiseless deduplication)
                  - Write to HDF5 (E3)
          6. Finalize HDF5, verify WAL consistency
        """
        t_start = time.time()
        sweep_result = SweepResult(sweep_id=self._config.sweep_id)
        errors: list[str] = []

        print(f"\n{'='*70}")
        print(f"  SWEEP ENGINE — {self._config.sweep_id}")
        print(f"{'='*70}")

        # ── Step 1: Validate ──
        validation_errors = validate_sweep_config(self._config)
        if validation_errors:
            for err in validation_errors:
                print(f"  [CONFIG ERROR] {err}")
            sweep_result.errors = validation_errors
            return sweep_result

        # ── Step 2: Load calibrations ──
        print("\n── Loading calibrations ──")
        for cal_path in self._config.calibrations:
            try:
                self._load_calibration(cal_path)
                print(f"  Loaded: {cal_path}")
            except Exception as e:
                msg = f"Failed to load calibration {cal_path}: {e}"
                print(f"  [ERROR] {msg}")
                errors.append(msg)

        if not self._cal_cache:
            errors.append("No calibrations loaded successfully")
            sweep_result.errors = errors
            return sweep_result

        # ── Step 3: Expand grid ──
        print("\n── Expanding sweep grid ──")
        tasks = expand_grid(self._config)
        self._progress.total_tasks = len(tasks)
        print(f"  Total tasks: {len(tasks)}")
        print(f"  Experiments: {len(self._config.experiments)}")
        print(f"  Calibrations: {len(self._cal_cache)}")

        if not tasks:
            errors.append("Grid expansion produced zero tasks")
            sweep_result.errors = errors
            return sweep_result

        # ── Step 4: Group tasks for cache locality ──
        groups = self._group_tasks(tasks)
        print(f"  Task groups: {len(groups)} "
              f"(hamiltonian × topology × calibration)")

        # ── Step 5: Open HDF5 and execute ──
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        hdf5_path = str(output_dir / self._config.hdf5_filename)

        sweep_attrs = {
            "sweep_id": self._config.sweep_id,
            "framework_version": self._config.framework_version,
            "total_tasks": len(tasks),
            "total_groups": len(groups),
            "calibrations": json.dumps(self._config.calibrations),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }

        print(f"\n── Executing sweep → {hdf5_path} ──")

        with SweepHDF5Writer(
            hdf5_path,
            sweep_attrs=sweep_attrs,
            enable_swmr=self._config.enable_swmr,
            debug_json=self._config.debug_json,
            debug_json_dir=str(output_dir / "debug_json") if self._config.debug_json else None,
        ) as writer:
            for group_idx, (group_key, group_tasks) in enumerate(groups.items()):
                ham_name, topo_name, cal_path = group_key
                print(f"\n  Group {group_idx + 1}/{len(groups)}: "
                      f"{ham_name} / {topo_name} / {_calibration_id(cal_path)}")

                try:
                    self._execute_group(
                        group_tasks, writer, errors,
                    )
                except Exception as e:
                    msg = f"Group {group_key} failed: {e}"
                    print(f"    [ERROR] {msg}")
                    errors.append(msg)

        # ── Step 6: Verify WAL consistency ──
        print("\n── Verifying WAL consistency ──")
        temp_writer = SweepHDF5Writer(hdf5_path)
        consistency = temp_writer.verify_consistency()
        print(f"  WAL entries: {consistency['wal_entries']}")
        print(f"  HDF5 groups: {consistency['hdf5_groups']}")
        print(f"  Consistent: {consistency['consistent']}")
        if not consistency["consistent"]:
            errors.append(
                f"WAL inconsistency: {consistency.get('missing_from_hdf5', 0)} "
                f"entries missing from HDF5"
            )

        # ── Summary ──
        elapsed = time.time() - t_start
        self._progress.elapsed_seconds = elapsed

        sweep_result.hdf5_path = hdf5_path
        sweep_result.total_tasks = self._progress.total_tasks
        sweep_result.total_placements = self._progress.total_placements
        sweep_result.total_simulations = self._progress.total_simulations
        sweep_result.total_deduplicated = self._progress.total_deduplicated
        sweep_result.total_hdf5_writes = self._progress.hdf5_writes
        sweep_result.total_errors = len(errors)
        sweep_result.elapsed_seconds = elapsed
        sweep_result.errors = errors

        print(f"\n{'='*70}")
        print(f"  SWEEP COMPLETE — {self._config.sweep_id}")
        print(f"  Tasks: {self._progress.completed_tasks}/{self._progress.total_tasks}")
        print(f"  Placements processed: {self._progress.total_placements}")
        print(f"  Simulations: {self._progress.total_simulations}")
        print(f"  Deduplicated: {self._progress.total_deduplicated}")
        print(f"  HDF5 writes: {self._progress.hdf5_writes}")
        print(f"  Errors: {len(errors)}")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  Output: {hdf5_path}")
        print(f"{'='*70}\n")

        return sweep_result

    # ── Internal: calibration loading ──

    def _load_calibration(self, cal_path: str) -> None:
        """Load and cache a calibration file, register device with solver."""
        if cal_path in self._cal_cache:
            return

        from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter

        adapter = IQMv2Adapter()
        device_cal = adapter.load(cal_path)

        # Load raw JSON for twin simulator noise model builder
        with open(cal_path) as f:
            cal_json = json.load(f)

        cal_id = _calibration_id(cal_path)
        self._cal_cache[cal_path] = (cal_id, cal_json, device_cal)

        # Register device with placement solver
        self._solver.add_device(device_cal)

    # ── Internal: task grouping ──

    def _group_tasks(
        self, tasks: list[SweepTask],
    ) -> dict[tuple[str, str, str], list[SweepTask]]:
        """Group tasks by (hamiltonian, topology, calibration).

        This grouping ensures placement solver results are reused
        across seeds within the same (topology, device) combination.
        """
        groups: dict[tuple[str, str, str], list[SweepTask]] = {}
        for task in tasks:
            key = (task.hamiltonian, task.topology_name, task.calibration_path)
            if key not in groups:
                groups[key] = []
            groups[key].append(task)
        return groups

    # ── Internal: group execution ──

    def _execute_group(
        self,
        tasks: list[SweepTask],
        writer: SweepHDF5Writer,
        errors: list[str],
    ) -> None:
        """Execute all tasks in a group.

        All tasks share the same (hamiltonian, topology, calibration).
        Placements are computed once and reused across seeds.
        """
        if not tasks:
            return

        representative = tasks[0]
        ham_name = representative.hamiltonian
        topo_name = representative.topology_name
        cal_path = representative.calibration_path
        qsize = representative.qubit_size

        # Resolve calibration
        cal_id, cal_json, device_cal = self._cal_cache[cal_path]

        # ── E1: Find placements ──
        cache_key = (topo_name, device_cal.device_id)
        if cache_key not in self._placement_cache:
            placements = self._solver.find_all_placements(
                circuit_edges=representative.topology_edges,
                circuit_qubits=qsize,
                device_ids=[device_cal.device_id],
                strategy="max_fidelity",
                max_placements=representative.max_placements,
            )
            self._placement_cache[cache_key] = placements
            print(f"    E1: {len(placements)} placements for {topo_name} "
                  f"on {device_cal.device_id}")
        else:
            placements = self._placement_cache[cache_key]
            print(f"    E1: {len(placements)} placements (cached)")

        if not placements:
            errors.append(
                f"No valid placements for {topo_name} on {device_cal.device_id}"
            )
            return

        self._progress.total_placements += len(placements) * len(tasks)

        # ── Build circuit and observable ──
        circuit, observable, ham_metadata = self._build_circuit_and_observable(
            ham_name, qsize, representative.topology_edges,
        )

        exact_energy = ham_metadata.get("exact_ground_energy")

        # NOTE (v1.1.1): Noiseless cache pre-population removed.
        # The old code ran run_twin_battery() in the parent process to
        # populate the noiseless dedup cache before forking workers.
        # This called AerSimulator in the parent, initializing C++ thread
        # pools whose mutexes deadlock child processes after fork.
        # See debug_e2_fork_order.py for proof.
        #
        # Workers now compute noiseless results independently. Cost:
        # N_workers × 2 noiseless sims × ~5ms = negligible for 4q.
        # The noiseless_cache is passed empty; each worker populates
        # its own copy. Dedup still works WITHIN each battery.

        # ── Collect work items ──
        work_items = []
        work_meta = []  # parallel metadata for result processing
        for task in tasks:
            seed = task.seed
            noise_envs = task.noise_configs

            for placement in placements:
                qubit_names = [
                    placement.qubit_mapping[i]
                    for i in range(qsize)
                ]
                placement_id_str = "_".join(qubit_names)

                work_items.append((
                    circuit, observable, qubit_names, cal_json, cal_id,
                    placement_id_str, placement.topology_hash, noise_envs,
                    seed * 1000 + placement.placement_id,
                    self._device, {},  # empty cache — workers compute independently
                ))
                work_meta.append((task, placement, qubit_names, seed))

        # ── Execute batteries ──
        cpu_workers = self._config.cpu_workers
        n_items = len(work_items)

        # E2: Route tasks through TieredExecutionPlanner for CPU/GPU decision
        from lumi_hpc_qc.sweep.execution_planner import SimulationTask, select_backend
        exec_backend = select_backend(qsize, method="density_matrix")
        print(f"    E2: Routing {n_items} batteries → {exec_backend} "
              f"({qsize}q, density_matrix)")

        if cpu_workers > 1 and n_items > 1:
            # E2/E7 Level 1: Parallel execution via multiprocessing
            # CRITICAL: Disable C++ thread pools before fork.
            # numpy BLAS (via exact_ground_energy → eigvalsh) and OpenMP
            # initialize thread pools whose mutexes deadlock children
            # after fork. Setting these to 1 prevents thread creation.
            # Parallelism comes from Pool workers, not per-worker BLAS.
            # See tests/debug_e2_fork_order.py for proof.
            import os
            os.environ.setdefault("OMP_NUM_THREADS", "1")
            os.environ.setdefault("MKL_NUM_THREADS", "1")
            os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
            os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

            actual_workers = min(self._planner.cpu_workers, n_items)
            print(f"    E2: Parallel execution — {n_items} batteries across "
                  f"{actual_workers} workers")
            t_par = time.time()

            # CRITICAL: Close HDF5 before fork. h5py.File holds C-level
            # HDF5 mutexes that deadlock forked children. Workers don't
            # use HDF5 — they return dicts. We reopen after Pool completes.
            writer.close()

            with mp.Pool(actual_workers) as pool:
                battery_results = pool.map(_battery_worker, work_items)

            # Reopen HDF5 for result writing (serial, after all workers done)
            writer.open()
            print(f"    E2: Parallel complete in {time.time()-t_par:.1f}s")
        else:
            # Serial fallback (single worker or single item)
            battery_results = [_battery_worker(item) for item in work_items]

        # ── Process results and write to HDF5 (serial — HDF5 not thread-safe) ──
        task_completion = set()
        for idx, br in enumerate(battery_results):
            task, placement, qubit_names, seed = work_meta[idx]
            placement_id_str = "_".join(qubit_names)

            if br["error"] is not None:
                self._progress.total_errors += 1
                errors.append(
                    f"Battery error: {task.task_id} / {placement_id_str}: "
                    f"{br['error']}"
                )
                continue

            battery = br["battery"]
            self._progress.total_simulations += battery.simulated_count
            self._progress.total_deduplicated += battery.deduplicated_count

            # ── E3: Write to HDF5 ──
            for twin_result in battery.results:
                if twin_result.error is not None:
                    self._progress.total_errors += 1
                    errors.append(
                        f"Sim error: {task.task_id} / {placement_id_str} / "
                        f"{twin_result.environment}: {twin_result.error}"
                    )
                    continue

                energy_val = twin_result.energy if twin_result.energy is not None else 0.0

                # Compute noise fingerprinting features from counts
                fingerprint = _compute_fingerprint(
                    twin_result.counts, qsize
                )

                # Extract per-edge CZ fidelity for this placement
                edge_fidelities = _extract_edge_fidelities(
                    placement, device_cal
                )

                entry = SweepResultEntry(
                    device_id=device_cal.device_id,
                    device_prefix=device_cal.device_prefix,
                    seed=seed,
                    placement_qubits=qubit_names,
                    calibration_id=cal_id,
                    noise_config=twin_result.environment,
                    energy_trajectory=[energy_val],
                    best_energy=energy_val,
                    total_iterations=1,
                    converged=True,
                    circuit_metrics={
                        "num_qubits": qsize,
                        "topology_name": topo_name,
                        "hamiltonian": ham_name,
                        "pre_transpilation_depth": circuit.depth(),
                    },
                    per_qubit_calibration=placement.per_qubit_calibration,
                    placement_score=placement.score,
                    topology_hash=placement.topology_hash,
                    wall_time_seconds=twin_result.execution_time_s,
                    framework_version=self._config.framework_version,
                    experiment_id=f"{self._config.sweep_id}_{task.task_id}_{placement_id_str}_{twin_result.environment}",
                    noise_fingerprint=fingerprint,
                    per_edge_cz_fidelity=edge_fidelities,
                )

                try:
                    writer.write(entry)
                    self._progress.hdf5_writes += 1
                except Exception as e:
                    errors.append(
                        f"HDF5 write error: {entry.group_path}: {e}"
                    )

            # Track task completion for progress
            if task.task_id not in task_completion:
                task_completion.add(task.task_id)
                self._progress.completed_tasks += 1
                if self._progress_callback:
                    self._progress_callback(self._progress)

        # Print group summary
        print(f"    Completed: {len(tasks)} seeds × {len(placements)} placements "
              f"= {self._progress.hdf5_writes} HDF5 writes "
              f"({self._progress.total_deduplicated} deduplicated)")

    # ── Internal: circuit building ──

    def _build_circuit_and_observable(
        self,
        hamiltonian_name: str,
        num_qubits: int,
        edges: list[tuple[int, int]],
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Build a reference circuit and observable for evaluation.

        For characterization sweeps, builds a simple measurement
        circuit from the hamiltonian via the plugin registry.

        The sweep YAML uses domain-language names ("tfim", "heisenberg",
        "fermi_hubbard") which map directly to plugin registry names.
        Shorthand aliases ("fh" → "fermi_hubbard") are resolved here.

        Returns:
            (circuit, observable, metadata_dict)
        """
        from qiskit import QuantumCircuit
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        from lumi_hpc_qc.types import ExperimentConfig

        registry = PluginRegistry()
        registry.discover()

        # Resolve aliases
        plugin_name = _SWEEP_TO_PLUGIN.get(hamiltonian_name, hamiltonian_name)

        # Build default model_params per plugin
        model_params = _default_model_params(plugin_name, num_qubits)

        exp_config = ExperimentConfig(
            model=plugin_name,
            model_params=model_params,
            ansatz="su2",
            ansatz_params={"reps": 1},
            optimizer="cobyla",
            optimizer_params={"maxiter": 1},
            gradient="none",
            initializer="random",
            initializer_params={"seed": 0},
            backend="aer_gpu",
            precision="double",
            num_qubits=num_qubits,
            mode="interactive",
            output_dir="/tmp/sweep_tmp",
        )

        ham_builder = registry.get_hamiltonian(plugin_name)
        hamiltonian, ham_meta = ham_builder.build(exp_config)

        exact_energy = None
        if num_qubits <= 24:
            try:
                exact_energy = ham_builder.exact_ground_energy(hamiltonian)
            except Exception:
                pass

        metadata = {
            "model": hamiltonian_name,
            "num_qubits": num_qubits,
            "num_pauli_terms": len(hamiltonian) if hasattr(hamiltonian, '__len__') else 0,
            "exact_ground_energy": exact_energy,
        }

        # For characterization, evaluate ⟨ψ₀|H|ψ₀⟩ where |ψ₀⟩ = |00...0⟩
        circuit = QuantumCircuit(num_qubits)

        return circuit, hamiltonian, metadata


# ═══════════════════════════════════════════════════════════════════════
# Plugin name resolution
# ═══════════════════════════════════════════════════════════════════════

# Map sweep YAML shorthand aliases → plugin registry names
_SWEEP_TO_PLUGIN: dict[str, str] = {
    "fh": "fermi_hubbard",
    "qaoa": "qaoa_maxcut",
}


def _default_model_params(plugin_name: str, num_qubits: int) -> dict[str, Any]:
    """Return sensible default model_params for a plugin.

    Each Hamiltonian plugin expects specific keys in model_params.
    These defaults produce a reasonable sweep Hamiltonian for the
    given qubit count. Researchers override via the YAML config
    when they need non-default couplings or lattice shapes.
    """
    if plugin_name == "tfim":
        return {
            "num_qubits": num_qubits,
            "j": 1.0,
            "g": 1.0,
            "boundary_condition": "open",
        }
    elif plugin_name == "heisenberg":
        return {
            "lattice_rows": 1,
            "lattice_cols": num_qubits,
            "jx": 1.0,
            "jy": 1.0,
            "jz": 1.0,
            "boundary_condition": "open",
        }
    elif plugin_name == "fermi_hubbard":
        return {
            "lattice_rows": 1,
            "lattice_cols": max(2, num_qubits // 2),
            "hopping_t": 1.0,
            "interaction_u": 2.0,
        }
    else:
        return {"num_qubits": num_qubits}


# ═══════════════════════════════════════════════════════════════════════
# Noise fingerprinting — computed during execution, stored in HDF5
# ═══════════════════════════════════════════════════════════════════════

def _compute_fingerprint(
    counts: dict[str, int] | None,
    num_qubits: int,
) -> dict[str, float | int | None]:
    """Compute noise fingerprinting features from measurement counts.

    Features F1, F2, F5, F6, F8 + original 4 (measurement_entropy,
    dominant_bitstring_fraction, num_unique_bitstrings,
    dominant_bitstring_hamming_weight).

    F3 (z_group_expectation_mean), F4 (xz_expectation_ratio), and
    F7 (expectation_variance_across_groups) require per-Pauli-group
    data from measurement stats — not available from raw counts alone.
    These are null here; populated when VQE measurement stats exist.

    Returns empty dict if counts is None (noiseless/statevector).
    """
    if not counts:
        return {}

    total = sum(counts.values())
    if total == 0:
        return {}

    # ── Original 4 ──
    # measurement_entropy: Shannon entropy
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)

    # dominant_bitstring_fraction
    max_count = max(counts.values())
    dbf = max_count / total

    # num_unique_bitstrings
    n_unique = len(counts)

    # ── F1: bitstring_hamming_weight_mean ──
    hw_sum = sum(bs.count("1") * cnt for bs, cnt in counts.items())
    hw_mean = hw_sum / total

    # ── F2: bitstring_hamming_weight_variance ──
    hw_var = sum(((bs.count("1") - hw_mean) ** 2) * cnt for bs, cnt in counts.items()) / total

    # ── F5: effective_hilbert_dimension (participation ratio) ──
    sum_p_sq = sum((c / total) ** 2 for c in counts.values())
    ehd = 1.0 / sum_p_sq if sum_p_sq > 0 else None

    # ── F6: kl_divergence_from_uniform ──
    n_states = 2 ** num_qubits
    q = 1.0 / n_states
    kl = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            kl += p * math.log2(p / q)

    # ── F8: dominant_bitstring_hamming_weight ──
    dominant_bs = max(counts, key=counts.get)
    dbhw = dominant_bs.count("1")

    return {
        "measurement_entropy": entropy,
        "dominant_bitstring_fraction": dbf,
        "num_unique_bitstrings": n_unique,
        "bitstring_hamming_weight_mean": hw_mean,
        "bitstring_hamming_weight_variance": hw_var,
        "z_group_expectation_mean": None,       # F3: needs Pauli-group data
        "xz_expectation_ratio": None,           # F4: needs Pauli-group data
        "effective_hilbert_dimension": ehd,
        "kl_divergence_from_uniform": kl,
        "expectation_variance_across_groups": None,  # F7: needs Pauli-group data
        "dominant_bitstring_hamming_weight": dbhw,
    }


def _extract_edge_fidelities(
    placement: Any,
    device_cal: Any,
) -> list[float]:
    """Extract per-edge CZ fidelity for a placement's internal edges.

    Returns a list of CZ fidelity values for each edge within the
    placement subgraph, ordered by (min_idx, max_idx).
    """
    idx_set = set(placement.physical_indices)
    fidelities = []
    for i in sorted(placement.physical_indices):
        for j in device_cal.adjacency.get(i, set()):
            if j in idx_set and j > i:
                fidelities.append(device_cal.gate_fidelity(i, j))
    return fidelities


# ═══════════════════════════════════════════════════════════════════════
# Convenience: run from YAML file
# ═══════════════════════════════════════════════════════════════════════

def run_sweep_from_yaml(yaml_path: str, *, device: str = "CPU") -> SweepResult:
    """Load a YAML config and run the complete sweep.

    Args:
        yaml_path: Path to the YAML sweep configuration file.
        device: "CPU" or "GPU" for AerSimulator.

    Returns:
        SweepResult with paths to output files and summary statistics.
    """
    import yaml

    with open(yaml_path) as f:
        yaml_dict = yaml.safe_load(f)

    config = parse_sweep_config(yaml_dict)

    engine = SweepEngine(config, device=device)
    return engine.run()


def run_sweep_from_dict(
    config_dict: dict[str, Any],
    *,
    device: str = "CPU",
    progress_callback: Any | None = None,
) -> SweepResult:
    """Run a sweep from a Python dict (for programmatic use and testing).

    Args:
        config_dict: Dict matching the YAML sweep schema.
        device: "CPU" or "GPU".
        progress_callback: Optional callback(SweepProgress) for live updates.

    Returns:
        SweepResult.
    """
    config = parse_sweep_config(config_dict)

    engine = SweepEngine(
        config,
        device=device,
        progress_callback=progress_callback,
    )
    return engine.run()
