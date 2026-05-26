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
import sys
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
class SamplingConfig:
    """LHS or other sampling configuration for parameter space exploration.

    When method="lhs", generates Latin Hypercube samples across the
    parameter ranges. Each sample becomes a SweepTask with unique
    model_params that override the Hamiltonian plugin's defaults.

    v1.2.0 Item C — RED-SPEC-003.
    """
    method: str = "grid"           # "grid" (default Cartesian) or "lhs"
    n_samples: int = 100           # number of LHS samples
    parameters: dict[str, list[float]] = field(default_factory=dict)
    # Each value is [min, max] range, e.g. {"j": [0.5, 2.0], "g": [0.5, 2.0]}
    seed: int = 42                 # LHS reproducibility seed


@dataclass
class QPUConfig:
    """QPU execution parameters — parsed from the ``qpu:`` YAML section.

    All behaviors default to OFF or safe values.  The researcher opts in
    to retry, timing capture, and queue prefetch explicitly.

    RED-DIRECTIVE-QPU-CONFIG-v1.0 §3.
    """
    shots: int = 4096
    batch_limit: int | None = None          # None = auto-detect from VTT API, fallback 100
    connection_timeout_s: int = 60

    # Retry policy — DISABLED by default (RED-DIRECTIVE-QPU-CONFIG §2 Finding 1)
    retry_enabled: bool = False
    retry_max_attempts: int = 3             # total attempts (1 = original only)
    retry_base_wait_s: float = 1.0          # doubles each attempt: 1s, 2s, 4s
    retry_errors: list[str] = field(default_factory=lambda: [
        "No results available", "timeout", "ConnectionError",
    ])

    # QX API features — OFF by default (Finding 4)
    timing_capture: bool = False
    queue_prefetch: bool = False


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
    seed_list: list[int] | None = None  # v1.4.0 — explicit seed list (overrides seeds/seed_offset)

    # Noise scope
    noise_configs: str | list[str] = "all"  # "all" or explicit list
    measurement_stats_interval_override: int | None = None  # Override per-env default

    # Placement strategy
    placement: str | int = "all_valid"  # "all_valid", "top_N", or int

    # VQE-specific grid (only for vqe_sweep type)
    grid: dict[str, list[Any]] = field(default_factory=dict)

    # LHS / parameter sampling (v1.2.0 Item C)
    sampling: SamplingConfig | None = None

    # BYO circuit (SPEC-002 §7.5) — used when experiment_type == "byo_circuit".
    # The factory's parameters are partitioned across grid (swept), fixed
    # (constant), and disorder (per-seed supplied data); see §7.5.1-§7.5.5.
    circuit_script: str = ""
    circuit_function: str = "build_circuit"
    fixed: dict[str, Any] = field(default_factory=dict)
    disorder: dict[str, Any] = field(default_factory=dict)
    signature_check: bool = True
    disorder_gates: list[str] = field(default_factory=lambda: ["rz", "rzz"])

    # Metadata
    label: str = ""


@dataclass
class PackingConfig:
    """Global pool packing parameters — parsed from ``sweep.packing`` YAML.

    Top-level config (not per-experiment) because global pool packing is
    inherently cross-experiment.

    v1.4.0 — RED-RESP-V140-DESIGN-v1.0 (REVISED) §2 Q1.
    """
    strategy: str = "dsatur"           # "dsatur" (default, current) or "global_pool"
    objective: str = "max_throughput"   # v1.4.0: only max_throughput implemented
    seed: int = 42                     # packing seed for deterministic assignment


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

    # QPU execution parameters (RED-DIRECTIVE-QPU-CONFIG-v1.0)
    qpu: QPUConfig = field(default_factory=QPUConfig)

    # Packing strategy (v1.4.0 — RED-RESP-V140-DESIGN §2 Q1)
    packing: PackingConfig = field(default_factory=PackingConfig)

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
    model_params: dict[str, float] = field(default_factory=dict)

    # BYO circuit (SPEC-002 §7.5). circuit_params is the per-task grid point —
    # kept SEPARATE from the Hamiltonian-routed model_params (§B-4). The build
    # seam assembles fixed_params ∪ disorder_instance ∪ circuit_params and
    # spreads them into the factory.
    circuit_params: dict[str, Any] = field(default_factory=dict)
    circuit_script: str = ""
    circuit_function: str = "build_circuit"
    fixed_params: dict[str, Any] = field(default_factory=dict)
    disorder_instance: dict[str, Any] = field(default_factory=dict)
    disorder_gates: tuple[str, ...] = ("rz", "rzz")
    # D3.4b: master_seed from the resolved disorder _meta, so the BYO counts run
    # can derive seed_simulator = resolve_instance_seed(master_seed, seed) —
    # identical to the banked floquet_runner_v2 (one seed per instance, driving
    # both disorder and shots). None -> entropy (not reproducible).
    master_seed: int | None = None


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
            # D3: "all" means the synthetic channel tiers (the historical 11),
            # NOT device_calibrated. device_calibrated is a different execution
            # path (statevector counts, D3.4) and must be requested by name, so
            # adding it to NOISE_ENVIRONMENTS does not silently inject it into
            # every existing "all" sweep.
            noise_envs = [e for e in NOISE_ENVIRONMENTS if e.source == "channels"]
        else:
            nc_names = exp.noise_configs if isinstance(exp.noise_configs, list) else [exp.noise_configs]
            # RED-RESP §7.5 F-6: never silently drop unknown noise-config names.
            # The previous `if n in NOISE_ENV_BY_NAME` filter dropped typos with
            # no error, so a sweep would quietly run a *different* set of
            # environments than the config requested -- a latent data-integrity
            # bug affecting all experiment types. Fail loud instead.
            unknown = [n for n in nc_names if n not in NOISE_ENV_BY_NAME]
            if unknown:
                raise ValueError(
                    f"Unknown noise config name(s) {unknown}. "
                    f"Available: {', '.join(NOISE_ENV_BY_NAME.keys())}"
                )
            noise_envs = [NOISE_ENV_BY_NAME[n] for n in nc_names]

        # Resolve seed values (v1.4.0 — seed_list overrides seeds/seed_offset)
        if exp.seed_list is not None:
            seed_values = exp.seed_list
        else:
            seed_values = [exp.seed_offset + i for i in range(exp.seeds)]

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

        # ── BYO circuit expansion (SPEC-002 §7.5): seed OUTER, grid INNER ──
        if exp.experiment_type == "byo_circuit":
            task_counter = _expand_byo_experiment(
                exp, config, seed_values, noise_envs,
                placement_strategy, max_placements, tasks, task_counter,
            )
            continue

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
                        # ── LHS sampling path (v1.2.0 Item C) ──
                        if (exp.sampling is not None
                                and exp.sampling.method == "lhs"
                                and exp.sampling.parameters):
                            lhs_samples = _generate_lhs_samples(exp.sampling)
                            for sample_idx, params in enumerate(lhs_samples):
                                for seed in seed_values:
                                    task_counter += 1
                                    tasks.append(SweepTask(
                                        task_id=f"T{task_counter:06d}",
                                        hamiltonian=ham,
                                        qubit_size=qsize,
                                        topology_name=topo_name,
                                        topology_edges=list(topo_spec["edges"]),
                                        seed=seed,
                                        calibration_path=cal_path,
                                        calibration_id=_calibration_id(cal_path),
                                        experiment_type=exp.experiment_type,
                                        noise_configs=noise_envs,
                                        placement_strategy=placement_strategy,
                                        max_placements=max_placements,
                                        label=exp.label,
                                        model_params=params,
                                    ))
                        else:
                            # ── Standard grid expansion ──
                            for seed in seed_values:
                                task_counter += 1
                                tasks.append(SweepTask(
                                    task_id=f"T{task_counter:06d}",
                                    hamiltonian=ham,
                                    qubit_size=qsize,
                                    topology_name=topo_name,
                                    topology_edges=list(topo_spec["edges"]),
                                    seed=seed,
                                    calibration_path=cal_path,
                                    calibration_id=_calibration_id(cal_path),
                                    experiment_type=exp.experiment_type,
                                    noise_configs=noise_envs,
                                    placement_strategy=placement_strategy,
                                    max_placements=max_placements,
                                    label=exp.label,
                                ))

    return tasks


def _expand_byo_experiment(
    exp: SweepExperimentConfig,
    config: SweepConfig,
    seed_values: list[int],
    noise_envs: list,
    placement_strategy: str,
    max_placements: int | None,
    tasks: list[SweepTask],
    task_counter: int,
) -> int:
    """Expand one ``byo_circuit`` experiment into tasks (SPEC-002 §7.5).

    Seed is the OUTER axis, the parameter grid the INNER axis. Per-seed disorder
    is resolved ONCE and the identical realization is attached to every grid
    point in that seed, so the cross-grid invariant is structural (§7.5.4). The
    factory signature is validated against grid ∪ fixed ∪ disorder keys, and a
    default-ON cross-grid identity check confirms the factory does not draw
    build-time randomness. Raises ValueError on any of these (submit-time,
    before execution), consistent with the F-6 fail-loud precedent.
    """
    # Local imports: these pull qiskit (circuit_loader); keep them off the
    # module import path so non-BYO sweeps don't pay for them.
    from lumi_hpc_qc.sweep.byo_sweep import (
        expand_circuit_grid, resolve_disorder, validate_factory_signature,
        cross_grid_identity_check,
    )
    from lumi_hpc_qc.sweep.circuit_loader import (
        load_factory, load_circuit, extract_disorder_signature,
    )

    if "num_qubits" not in exp.fixed:
        raise ValueError("byo_circuit requires fixed.num_qubits")
    num_qubits = int(exp.fixed["num_qubits"])

    grid_points = expand_circuit_grid(exp.grid)

    # Resolve per-seed disorder once (file load is RNG-free; generate is a
    # serial pre-pass). initial_state, if present, lives in the disorder block.
    resolved_disorder, _disorder_meta = resolve_disorder(
        exp.disorder, seed_values,
        num_qubits=num_qubits,
        configured_initial_state=exp.disorder.get("initial_state"),
    )
    # D3.4b: carry the disorder's master_seed onto each task so the counts run
    # derives seed_simulator = resolve_instance_seed(master_seed, seed). For
    # source=file the meta echoes the file's _meta.master_seed; for generate it
    # is the spec's master_seed. None (absent) -> entropy / not reproducible.
    disorder_master_seed = _disorder_meta.get("master_seed")

    # Signature check against the actual disorder field names (§7.5.1, F3).
    disorder_keys: set[str] = set()
    if resolved_disorder:
        disorder_keys = set(next(iter(resolved_disorder.values())).keys())
    factory = load_factory(exp.circuit_script, exp.circuit_function)
    validate_factory_signature(
        factory,
        grid_keys=set(exp.grid),
        fixed_keys=set(exp.fixed),
        disorder_keys=disorder_keys,
        allow_kwargs=not exp.signature_check,
    )

    # Default-ON cross-grid disorder-identity backstop (§7.5.4). Build the min
    # and max grid points for one representative seed and confirm the
    # disorder-bearing gates are identical — impurity (build-time RNG) is a
    # property of the factory, so one seed suffices.
    if len(grid_points) >= 2 and resolved_disorder:
        gate_names = tuple(exp.disorder_gates)
        primary_axis = next(iter(exp.grid), None)
        rep_instance = resolved_disorder[seed_values[0]]

        def _build(**kw):
            return load_circuit(
                script_file=exp.circuit_script,
                script_function=exp.circuit_function,
                script_params=kw,
            ).circuit

        cross_grid_identity_check(
            _build,
            fixed=exp.fixed,
            instance=rep_instance,
            grid_points=grid_points,
            extract_disorder_params=lambda qc: extract_disorder_signature(qc, gate_names),
            primary_axis=primary_axis,
        )

    # Expand: cal × seed (OUTER) × grid point (INNER).
    gate_names = tuple(exp.disorder_gates)
    for cal_path in config.calibrations:
        for seed in seed_values:
            instance = resolved_disorder[seed]
            for grid_point in grid_points:
                task_counter += 1
                tasks.append(SweepTask(
                    task_id=f"T{task_counter:06d}",
                    qubit_size=num_qubits,
                    seed=seed,
                    calibration_path=cal_path,
                    calibration_id=_calibration_id(cal_path),
                    experiment_type="byo_circuit",
                    noise_configs=noise_envs,
                    placement_strategy=placement_strategy,
                    max_placements=max_placements,
                    label=exp.label,
                    circuit_params=dict(grid_point),
                    circuit_script=exp.circuit_script,
                    circuit_function=exp.circuit_function,
                    fixed_params=dict(exp.fixed),
                    disorder_instance=instance,
                    disorder_gates=gate_names,
                    master_seed=disorder_master_seed,
                ))
    return task_counter


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


def _parse_seed_range(range_str: str) -> list[int]:
    """Parse a seed range string into a list of integers.

    Accepts comma-separated tokens where each token is either a single
    integer or a dash-separated range (inclusive on both ends).

    v1.4.0 — ``seed_list`` support (COMMS-021, RED-RESP-V140 §2 Q2).

    Examples:
        >>> _parse_seed_range("0-4,10-14,42")
        [0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 42]
        >>> _parse_seed_range("5")
        [5]
    """
    seeds: list[int] = []
    for token in range_str.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            parts = token.split("-", 1)
            lo, hi = int(parts[0]), int(parts[1])
            seeds.extend(range(lo, hi + 1))
        else:
            seeds.append(int(token))
    return seeds


def _generate_lhs_samples(
    sampling: SamplingConfig,
) -> list[dict[str, float]]:
    """Generate Latin Hypercube samples scaled to parameter ranges.

    Uses scipy.stats.qmc.LatinHypercube for quasi-random sampling
    that provides better coverage of high-dimensional parameter spaces
    than grid or random sampling with the same number of points.

    v1.2.0 Item C — RED-SPEC-003.

    Args:
        sampling: SamplingConfig with method="lhs", parameter ranges,
                  n_samples, and seed.

    Returns:
        List of dicts, each mapping parameter name → sampled value.
    """
    from scipy.stats.qmc import LatinHypercube

    param_names = list(sampling.parameters.keys())
    param_ranges = list(sampling.parameters.values())
    d = len(param_names)

    if d == 0:
        return [{}]

    sampler = LatinHypercube(d=d, seed=sampling.seed)
    raw = sampler.random(n=sampling.n_samples)  # shape (n_samples, d) in [0, 1]

    # Scale from [0, 1] to [lo, hi] for each parameter
    samples = []
    for row in raw:
        params: dict[str, float] = {}
        for i, name in enumerate(param_names):
            lo, hi = param_ranges[i][0], param_ranges[i][1]
            params[name] = lo + row[i] * (hi - lo)
        samples.append(params)

    return samples


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
            circuit_script=exp_dict.get("circuit_script", ""),
            circuit_function=exp_dict.get("circuit_function", "build_circuit"),
            fixed=exp_dict.get("fixed", {}),
            disorder=exp_dict.get("disorder", {}),
            signature_check=exp_dict.get("signature_check", True),
            disorder_gates=exp_dict.get("disorder_gates", ["rz", "rzz"]),
        )

        # Parse seed_list (v1.4.0 — explicit seed list)
        raw_seeds = exp_dict.get("seed_list")
        if raw_seeds is not None:
            if isinstance(raw_seeds, list):
                exp.seed_list = [int(s) for s in raw_seeds]
            elif isinstance(raw_seeds, str):
                exp.seed_list = _parse_seed_range(raw_seeds)
            elif isinstance(raw_seeds, int):
                exp.seed_list = [raw_seeds]

        # Parse sampling block (v1.2.0 Item C)
        sampling_dict = exp_dict.get("sampling")
        if sampling_dict and isinstance(sampling_dict, dict):
            # Convert parameter ranges: {"j": [0.5, 2.0]} stays as-is
            params_raw = sampling_dict.get("parameters", {})
            params = {}
            for k, v in params_raw.items():
                if isinstance(v, list) and len(v) == 2:
                    params[k] = [float(v[0]), float(v[1])]
            exp.sampling = SamplingConfig(
                method=sampling_dict.get("method", "grid"),
                n_samples=int(sampling_dict.get("n_samples", 100)),
                parameters=params,
                seed=int(sampling_dict.get("seed", 42)),
            )

        experiments.append(exp)

    # Parse execution section
    execution = sweep_dict.get("execution", {})

    # Parse QPU config section (RED-DIRECTIVE-QPU-CONFIG-v1.0 §3)
    qpu_dict = sweep_dict.get("qpu", {})
    retry_dict = qpu_dict.get("retry", {})
    qpu_config = QPUConfig(
        shots=int(qpu_dict.get("shots", 4096)),
        batch_limit=int(qpu_dict["batch_limit"]) if "batch_limit" in qpu_dict else None,
        connection_timeout_s=int(qpu_dict.get("connection_timeout_s", 60)),
        retry_enabled=bool(retry_dict.get("enabled", False)),
        retry_max_attempts=int(retry_dict.get("max_attempts", 3)),
        retry_base_wait_s=float(retry_dict.get("base_wait_s", 1.0)),
        retry_errors=retry_dict.get("retryable_errors", [
            "No results available", "timeout", "ConnectionError",
        ]),
        timing_capture=bool(qpu_dict.get("timing_capture", False)),
        queue_prefetch=bool(qpu_dict.get("queue_prefetch", False)),
    )

    # Parse packing config (v1.4.0 — RED-RESP-V140-DESIGN §2 Q1)
    packing_dict = sweep_dict.get("packing", {})
    packing_config = PackingConfig(
        strategy=str(packing_dict.get("strategy", "dsatur")),
        objective=str(packing_dict.get("objective", "max_throughput")),
        seed=int(packing_dict.get("seed", 42)),
    )

    config = SweepConfig(
        experiments=experiments,
        calibrations=sweep_dict.get("calibrations", []),
        cpu_workers=execution.get("cpu_workers", 128),
        gpu_workers=execution.get("gpu_workers", 8),
        output_dir=sweep_dict.get("output_dir", "sweep_output"),
        hdf5_filename=sweep_dict.get("hdf5_filename", "sweep.h5"),
        enable_swmr=sweep_dict.get("enable_swmr", False),
        debug_json=sweep_dict.get("debug_json", False),
        qpu=qpu_config,
        packing=packing_config,
        sweep_id=sweep_dict.get("sweep_id", str(uuid.uuid4())[:8]),
        framework_version=sweep_dict.get("framework_version", _pkg_version),
    )

    return config


def _validate_byo_experiment(prefix: str, exp: SweepExperimentConfig) -> list[str]:
    """Cheap, pre-submit structural checks for a byo_circuit experiment.

    The heavy checks (factory signature against the resolved disorder keys, and
    the cross-grid identity backstop) run in ``_expand_byo_experiment`` and
    raise; here we surface the fast structural problems as collected errors so
    the config-error path reports them cleanly with everything else.
    """
    errors: list[str] = []
    if not exp.circuit_script:
        errors.append(f"{prefix}: byo_circuit requires 'circuit_script'")
    elif not Path(exp.circuit_script).exists():
        errors.append(f"{prefix}: circuit_script not found: {exp.circuit_script}")
    if "num_qubits" not in exp.fixed:
        errors.append(f"{prefix}: byo_circuit requires fixed.num_qubits")
    if not exp.disorder:
        errors.append(
            f"{prefix}: byo_circuit requires a 'disorder' block (source: file|generate)"
        )
    elif exp.disorder.get("source", "file") == "file" and not exp.disorder.get("file"):
        errors.append(
            f"{prefix}: byo_circuit disorder source 'file' requires 'disorder.file'"
        )
    return errors


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
        if exp.experiment_type == "byo_circuit":
            # RED-RESP §3.1: byo_circuit is exempt from hamiltonians/qubit_sizes;
            # num_qubits comes from `fixed`. Validate the BYO fields instead.
            errors.extend(_validate_byo_experiment(prefix, exp))
        else:
            if not exp.hamiltonians:
                errors.append(f"{prefix}: no hamiltonians specified")
            if not exp.qubit_sizes:
                errors.append(f"{prefix}: no qubit_sizes specified")
        if exp.seeds < 1:
            errors.append(f"{prefix}: seeds must be >= 1, got {exp.seeds}")

        # Validate seed_list (v1.4.0)
        if exp.seed_list is not None:
            if len(exp.seed_list) == 0:
                errors.append(f"{prefix}: seed_list is empty")
            if any(s < 0 for s in exp.seed_list):
                errors.append(f"{prefix}: seed_list contains negative values")
            if len(exp.seed_list) != len(set(exp.seed_list)):
                errors.append(f"{prefix}: seed_list contains duplicates")

        # Validate noise config names. RED-RESP §7.5 F-6: also the scalar form
        # (e.g. `noise_configs: bogus`), which the list-only check missed and
        # expand_grid would otherwise have silently dropped.
        if exp.noise_configs not in ("all", ["all"]):
            nc_names = (
                exp.noise_configs if isinstance(exp.noise_configs, list)
                else [exp.noise_configs]
            )
            for nc_name in nc_names:
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

    # Validate packing config (v1.4.0)
    valid_strategies = {"dsatur", "global_pool"}
    if config.packing.strategy not in valid_strategies:
        errors.append(
            f"Unknown packing strategy '{config.packing.strategy}'. "
            f"Available: {sorted(valid_strategies)}"
        )

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


def _run_pool_subprocess(
    work_items: list[tuple],
    n_workers: int,
    project_dir: str,
    noiseless_cache: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run _battery_worker pool in a clean subprocess.

    The parent process has C++ thread pools (numpy BLAS, h5py HDF5,
    Aer) whose mutexes deadlock forked children at high worker counts.
    This function serializes work items, launches a fresh Python process
    that creates mp.Pool without inherited C++ state, and reads results
    back via pickle.

    Proven on LUMI: Pool(100) → 100/100 in 7.43s
    See tests/fork_test_subprocess.py

    Args:
        work_items: list of tuples matching _battery_worker signature
        n_workers: number of parallel workers
        project_dir: project root for sys.path in subprocess
        noiseless_cache: pre-computed noiseless results for dedup
            (from _precompute_noiseless_subprocess). If provided,
            injected into each worker's noiseless_cache argument.

    Returns:
        List of dicts with battery results (same as _battery_worker output)
    """
    import pickle
    import subprocess
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="hpcqc_pool_")
    items_path = os.path.join(tmp_dir, "work_items.pkl")
    results_path = os.path.join(tmp_dir, "results.pkl")

    # Serialize work items
    with open(items_path, "wb") as f:
        pickle.dump(work_items, f)

    # Serialize noiseless cache (if provided)
    cache_path = os.path.join(tmp_dir, "noiseless_cache.pkl")
    if noiseless_cache:
        with open(cache_path, "wb") as f:
            pickle.dump(noiseless_cache, f)

    # Write the subprocess runner script
    runner_path = os.path.join(tmp_dir, "pool_runner.py")
    has_cache = "True" if noiseless_cache else "False"
    with open(runner_path, "w") as f:
        f.write(f'''import sys, os, pickle, time, multiprocessing as mp
sys.path.insert(0, os.path.join("{project_dir}", "src"))
os.environ["OMP_NUM_THREADS"] = "1"

from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery

# Load pre-computed noiseless cache (if available)
noiseless_cache = {{}}
if {has_cache}:
    try:
        with open("{cache_path}", "rb") as f:
            noiseless_cache = pickle.load(f)
    except Exception:
        noiseless_cache = {{}}

def battery_worker(args):
    (circuit, observable, qubit_names, cal_json, cal_id,
     placement_id_str, topology_hash, noise_envs, seed,
     device, _empty_cache) = args
    try:
        battery = run_twin_battery(
            circuit=circuit, observable=observable,
            qubit_names=qubit_names, calibration_data=cal_json,
            calibration_id=cal_id, placement_id=placement_id_str,
            topology_hash=topology_hash, environments=noise_envs,
            seed=seed, device=device, noiseless_cache=dict(noiseless_cache),
        )
        return {{"placement_id_str": placement_id_str, "battery": battery, "error": None}}
    except Exception as e:
        return {{"placement_id_str": placement_id_str, "battery": None, "error": str(e)}}

with open("{items_path}", "rb") as f:
    work_items = pickle.load(f)

workers = min({n_workers}, len(work_items))
with mp.Pool(workers) as pool:
    results = pool.map(battery_worker, work_items)

with open("{results_path}", "wb") as f:
    pickle.dump(results, f)
''')

    # Launch clean subprocess
    result = subprocess.run(
        [sys.executable, runner_path],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Pool subprocess failed (rc={result.returncode}):\n"
            f"stderr: {result.stderr[:2000]}"
        )

    # Read results back
    with open(results_path, "rb") as f:
        battery_results = pickle.load(f)

    # Cleanup temp files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return battery_results


def _precompute_noiseless_subprocess(
    representative_items: list[tuple],
    project_dir: str,
) -> dict[str, Any]:
    """Pre-compute noiseless results in a clean subprocess.

    Runs one representative work item per unique (observable, topology)
    group, executing only noiseless-tier environments. Returns a cache
    dict that can be passed to the main Pool subprocess so workers skip
    redundant noiseless computations.

    Noiseless energy depends on observable + topology (for routing) but
    NOT on placement or seed. One computation per topology group serves
    all placements × all seeds.

    Args:
        representative_items: list of tuples, one per topology group.
            Same format as _battery_worker args but with noiseless-only
            noise_envs.
        project_dir: project root for sys.path in subprocess.

    Returns:
        Dict of cache_key → TwinResult for noiseless dedup.
        Empty dict on failure (graceful fallback to v1.1.1 behavior).
    """
    if not representative_items:
        return {}

    import pickle
    import subprocess
    import tempfile

    tmp_dir = tempfile.mkdtemp(prefix="hpcqc_noiseless_")
    items_path = os.path.join(tmp_dir, "noiseless_items.pkl")
    cache_path = os.path.join(tmp_dir, "noiseless_cache.pkl")

    with open(items_path, "wb") as f:
        pickle.dump(representative_items, f)

    runner_path = os.path.join(tmp_dir, "noiseless_runner.py")
    with open(runner_path, "w") as f:
        f.write(f'''import sys, os, pickle, hashlib
sys.path.insert(0, os.path.join("{project_dir}", "src"))
os.environ["OMP_NUM_THREADS"] = "1"

from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery

with open("{items_path}", "rb") as f:
    items = pickle.load(f)

cache = {{}}
for item in items:
    (circuit, observable, qubit_names, cal_json, cal_id,
     placement_id_str, topology_hash, noise_envs, seed,
     device, _empty_cache) = item
    try:
        battery = run_twin_battery(
            circuit=circuit, observable=observable,
            qubit_names=qubit_names, calibration_data=cal_json,
            calibration_id=cal_id, placement_id=placement_id_str,
            topology_hash=topology_hash, environments=noise_envs,
            seed=seed, device=device, noiseless_cache={{}},
        )
        # Extract cache entries from the battery results
        obs_hash = hashlib.sha256(str(observable).encode()).hexdigest()[:12]
        for result in battery.results:
            if result.error is None:
                key = f"{{obs_hash}}:{{circuit.num_qubits}}:{{topology_hash}}:{{result.environment}}"
                cache[key] = result
    except Exception as e:
        print(f"  Warning: noiseless precompute failed for {{topology_hash}}: {{e}}")

with open("{cache_path}", "wb") as f:
    pickle.dump(cache, f)
''')

    result = subprocess.run(
        [sys.executable, runner_path],
        capture_output=True, text=True,
    )

    if result.returncode != 0:
        # Graceful fallback — workers compute noiseless independently
        print(f"    Warning: noiseless precompute subprocess failed "
              f"(rc={result.returncode}). Falling back to per-worker compute.")
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)
        return {}

    with open(cache_path, "rb") as f:
        cache = pickle.load(f)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    return cache


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

        # Project root for subprocess worker scripts
        self._project_dir = os.environ.get(
            "PROJECT_DIR",
            os.environ.get(
                "SINGULARITYENV_PROJECT_DIR",
                str(Path(__file__).resolve().parents[3]),  # src/lumi_hpc_qc/sweep → root
            ),
        )

        # Plugin registry (shared across all methods, discovered once)
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        self._registry = PluginRegistry()
        self._registry.discover()

        # Timing accumulators for sweep_timing.json (v1.2.3)
        self._timing: dict[str, float] = {
            "placement_solving_s": 0.0,
            "circuit_build_s": 0.0,
            "noiseless_precompute_s": 0.0,
            "parallel_execution_s": 0.0,
            "hdf5_writes_s": 0.0,
        }

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
        t_phase = time.perf_counter()  # v1.2.3 timing harness
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

        t_config = time.perf_counter()  # end config parse

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

        t_cal = time.perf_counter()  # end calibration load

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

        t_grid = time.perf_counter()  # end grid expansion

        # ── Step 3b: Campaign manifest — resume or create (Item 6) ──
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "campaign_manifest.json"

        from lumi_hpc_qc.sweep.campaign_manifest import CampaignManifest
        from lumi_hpc_qc import __version__ as _fw_version

        if manifest_path.exists():
            self._manifest = CampaignManifest.load(manifest_path)
            completed = set(self._manifest.completed_tasks())
            original_count = len(tasks)
            tasks = [t for t in tasks if t.task_id not in completed]
            self._progress.total_tasks = len(tasks)
            print(f"  Resuming: {len(completed)} tasks completed, "
                  f"{len(tasks)} remaining (of {original_count})")
        else:
            task_ids = [t.task_id for t in tasks]
            self._manifest = CampaignManifest.create(
                campaign_id=self._config.sweep_id,
                task_ids=task_ids,
                framework_version=_fw_version,
            )
            self._manifest.save(manifest_path)
            print(f"  Manifest created: {len(task_ids)} tasks")

        if not tasks:
            print("  All tasks already completed — nothing to do")
            sweep_result.errors = errors
            return sweep_result

        # ── Step 4: Group tasks for cache locality ──
        groups = self._group_tasks(tasks)
        print(f"  Task groups: {len(groups)} "
              f"(hamiltonian × topology × calibration)")

        # ── Step 5: Open HDF5 and execute ──
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
                ham_name, topo_name, cal_path, _params_key = group_key
                print(f"\n  Group {group_idx + 1}/{len(groups)}: "
                      f"{ham_name} / {topo_name} / {_calibration_id(cal_path)}")

                try:
                    self._execute_group(
                        group_tasks, writer, errors,
                    )
                    # Item 6: Mark group tasks completed in manifest
                    for t in group_tasks:
                        self._manifest.mark_batch_completed(
                            batch_id=f"group_{group_idx}",
                            task_ids=[t.task_id],
                        )
                    self._manifest.save(manifest_path)
                except Exception as e:
                    msg = f"Group {group_key} failed: {e}"
                    print(f"    [ERROR] {msg}")
                    errors.append(msg)
                    # Item 6: Mark group tasks failed in manifest
                    for t in group_tasks:
                        self._manifest.mark_batch_failed(
                            batch_id=f"group_{group_idx}",
                            task_ids=[t.task_id],
                            error=str(e),
                        )
                    self._manifest.save(manifest_path)

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

        # ── Write sweep_timing.json (v1.2.3 — RED-DIRECTIVE-V123) ──
        self._write_timing_json(
            output_dir=str(output_dir),
            hdf5_path=hdf5_path,
            total_elapsed=elapsed,
            t_phase=t_phase,
            t_config=t_config,
            t_cal=t_cal,
            t_grid=t_grid,
        )

        # ── Write sweep_benchmark.parquet (v1.3.0 — RED-DIRECTIVE-V130 Item 1) ──
        try:
            from lumi_hpc_qc.data.benchmark_export import (
                export_benchmark_to_parquet,
                make_simulator_timing_records,
            )

            benchmark_path = output_dir / "sweep_benchmark.parquet"
            mode = "simulator"
            timing_records = []

            # Check if QPU backend has timing records
            backend = getattr(self, "_backend", None)
            if backend is not None and hasattr(backend, "get_batch_timings"):
                timing_records = backend.get_batch_timings()
                if timing_records:
                    mode = "qpu"
                    # Inject retry_attempts from backend (v1.3.1 Finding 7)
                    if hasattr(backend, "get_batch_retry_attempts"):
                        retry_attempts = backend.get_batch_retry_attempts()
                        for idx, rec in enumerate(timing_records):
                            ra = retry_attempts[idx] if idx < len(retry_attempts) else None
                            if isinstance(rec, dict):
                                rec["retry_attempts"] = ra
                            else:
                                try:
                                    rec.retry_attempts = ra
                                except AttributeError:
                                    pass  # dataclass without field — stays None in export

            # Fallback: simulator timing from sweep_timing.json
            if not timing_records:
                timing_json_path = Path(str(output_dir)) / "sweep_timing.json"
                if timing_json_path.exists():
                    with open(timing_json_path) as f:
                        timing_json = json.load(f)
                    timing_records = make_simulator_timing_records(timing_json)

            if timing_records:
                n_rows = export_benchmark_to_parquet(
                    timing_records=timing_records,
                    sweep_metadata={
                        "sweep_id": self._config.sweep_id,
                        "framework_version": self._config.framework_version,
                        "mode": mode,
                        "device": self._device,
                        "partition": os.getenv("SLURM_JOB_PARTITION", "unknown"),
                    },
                    output_path=benchmark_path,
                )
                print(f"  Benchmark Parquet: {n_rows} rows → {benchmark_path}")
        except Exception as e:
            print(f"  Benchmark Parquet: skipped ({e})")

        return sweep_result

    # ── Internal: calibration loading ──

    def _write_timing_json(
        self,
        output_dir: str,
        hdf5_path: str,
        total_elapsed: float,
        t_phase: float,
        t_config: float,
        t_cal: float,
        t_grid: float,
    ) -> None:
        """Write sweep_timing.json alongside sweep.h5.

        Captures per-phase timing, parallelism metrics, sampling config,
        Lustre storage context, and environment metadata.

        v1.2.3 — RED-DIRECTIVE-V123.
        """
        import subprocess as _sp

        # Sampling config (absent for grid-mode)
        sampling_info = None
        for exp in self._config.experiments:
            if hasattr(exp, "sampling") and exp.sampling and exp.sampling.method == "lhs":
                sampling_info = {
                    "method": "lhs",
                    "n_samples": exp.sampling.n_samples,
                    "n_parameters": len(exp.sampling.parameters),
                }
                break
        if sampling_info is None and any(
            hasattr(exp, "sampling") and exp.sampling
            for exp in self._config.experiments
        ):
            sampling_info = {"method": "grid"}

        # Parallelism
        total_tasks = self._progress.total_tasks
        workers = self._config.cpu_workers
        tasks_per_worker = total_tasks / workers if workers > 0 else 0
        par_exec = self._timing["parallel_execution_s"]
        # Estimate sequential time from per-task median
        total_sims = self._progress.total_simulations
        if total_sims > 0 and par_exec > 0:
            # Sequential estimate: if all tasks ran serially
            # Using total_sims * (par_exec / total_sims * workers) as estimate
            sequential_estimate = par_exec * workers
            parallel_efficiency = round(sequential_estimate / par_exec / workers, 2) if par_exec > 0 else None
        else:
            parallel_efficiency = None

        # Lustre storage context
        storage = {}
        container_path = os.environ.get("HPCQC_CPU_CONTAINER", "")
        stripe = self._get_stripe_info(hdf5_path)
        if stripe:
            storage["hdf5_path"] = hdf5_path
            storage.update({f"hdf5_{k}": v for k, v in stripe.items()})
        if container_path:
            container_stripe = self._get_stripe_info(container_path)
            if container_stripe:
                storage["container_path"] = container_path
                storage.update({f"container_{k}": v for k, v in container_stripe.items()})

        timing_data = {
            "sweep_id": self._config.sweep_id,
            "framework_version": self._config.framework_version,
            "total_elapsed_s": round(total_elapsed, 2),
            "phases": {
                "config_parse_s": round(t_config - t_phase, 2),
                "calibration_load_s": round(t_cal - t_config, 2),
                "grid_expansion_s": round(t_grid - t_cal, 2),
                "placement_solving_s": round(self._timing["placement_solving_s"], 2),
                "noiseless_precompute_s": round(self._timing["noiseless_precompute_s"], 2),
                "parallel_execution_s": round(self._timing["parallel_execution_s"], 2),
                "hdf5_writes_s": round(self._timing["hdf5_writes_s"], 2),
                "circuit_build_s": round(self._timing["circuit_build_s"], 2),
            },
            "parallelism": {
                "workers": workers,
                "tasks": total_tasks,
                "tasks_per_worker": round(tasks_per_worker, 1),
                "parallel_efficiency": parallel_efficiency,
            },
            "environment": {
                "node": os.environ.get("SLURMD_NODENAME", os.environ.get("HOSTNAME", "unknown")),
                "partition": os.environ.get("SLURM_JOB_PARTITION", "unknown"),
                "cpus": int(os.environ.get("SLURM_CPUS_ON_NODE", os.environ.get("SLURM_CPUS_PER_TASK", "0"))),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
                "container": os.path.basename(container_path) if container_path else "",
            },
        }

        if sampling_info:
            timing_data["sampling"] = sampling_info
        if storage:
            timing_data["storage"] = storage

        timing_path = os.path.join(output_dir, "sweep_timing.json")
        try:
            with open(timing_path, "w") as f:
                json.dump(timing_data, f, indent=2)
        except Exception:
            pass  # timing is diagnostic — never fail a sweep

    @staticmethod
    def _get_stripe_info(path: str) -> dict:
        """Get Lustre stripe info via lfs getstripe. Returns {} if unavailable."""
        import subprocess as _sp
        try:
            result = _sp.run(
                ["lfs", "getstripe", path],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode != 0:
                return {}
            info = {}
            for line in result.stdout.splitlines():
                line = line.strip()
                if "lmm_stripe_count:" in line:
                    info["stripe_count"] = int(line.split()[-1])
                elif "lmm_stripe_size:" in line:
                    info["stripe_size_bytes"] = int(line.split()[-1])
            return info
        except (FileNotFoundError, _sp.TimeoutExpired, ValueError):
            return {}

    def _load_calibration(self, cal_path: str) -> None:
        """Load and cache a calibration file, register device with solver.

        Reads the 'adapter' field from the calibration JSON to determine
        which calibration adapter to use. Defaults to 'iqm_v2' for
        backward compatibility with files that lack the field.

        v1.2.1 — RED-DIRECTIVE-V121 Items 1+4.
        """
        if cal_path in self._cal_cache:
            return

        # Load raw JSON — used for adapter routing AND twin sim noise model
        with open(cal_path) as f:
            cal_json = json.load(f)

        # Item 4: explicit adapter field, default iqm_v2 for backward compat
        adapter_name = cal_json.get("adapter", "iqm_v2")
        adapter = self._registry.get_calibration_adapter(adapter_name)
        device_cal = adapter.load(cal_path)

        cal_id = _calibration_id(cal_path)
        self._cal_cache[cal_path] = (cal_id, cal_json, device_cal)

        # Register device with placement solver
        self._solver.add_device(device_cal)

    # ── Internal: task grouping ──

    def _group_tasks(
        self, tasks: list[SweepTask],
    ) -> dict[tuple[str, str, str, tuple], list[SweepTask]]:
        """Group tasks by (hamiltonian, topology, calibration, model_params).

        This grouping ensures placement solver results are reused
        across seeds within the same (topology, device) combination.
        Tasks with different model_params (e.g. from LHS sampling) are
        placed in separate groups since they produce different Hamiltonians.
        """
        groups: dict[tuple[str, str, str, tuple], list[SweepTask]] = {}
        for task in tasks:
            params_key = tuple(sorted(task.model_params.items())) if task.model_params else ()
            if task.experiment_type == "byo_circuit":
                # D3.4: BYO tasks have no hamiltonian/topology_name; each
                # circuit_script is a distinct circuit family whose placements
                # come from the BUILT circuit's connectivity (not topology_edges).
                # Group by (script, calibration) so a script's placement solve is
                # reused across seeds/grid-points. Keep the 4-tuple shape so the
                # run() unpack (ham,topo,cal,params) stays valid — script goes in
                # the hamiltonian slot, "byo" in the topology slot for logging.
                key = (task.circuit_script, "byo", task.calibration_path, params_key)
            else:
                key = (task.hamiltonian, task.topology_name, task.calibration_path, params_key)
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

        # D3.4: BYO circuits use a separate counts-based execution path
        # (placements from the built circuit's connectivity, device_calibrated
        # statevector, counts -> autocorrelator), not the hamiltonian/⟨H⟩ twin
        # battery. Dispatch before any hamiltonian/topology work.
        if tasks[0].experiment_type == "byo_circuit":
            self._execute_byo_group(tasks, writer, errors)
            return

        # A device_calibrated (source != "channels") env must not reach the
        # synthetic-channel twin battery — that would run device-calibrated noise
        # through the density_matrix channel path (wrong physics). It belongs on
        # the BYO counts path above. Refuse loudly if it lands here.
        bad_source = sorted({
            e.source for t in tasks for e in t.noise_configs
            if e.source != "channels"
        })
        if bad_source:
            raise NotImplementedError(
                f"noise source(s) {bad_source} (e.g. 'device_calibrated') are "
                f"only valid on the BYO counts path (experiment_type=byo_circuit). "
                f"The synthetic-channel battery cannot run them."
            )

        representative = tasks[0]
        ham_name = representative.hamiltonian
        topo_name = representative.topology_name
        cal_path = representative.calibration_path
        qsize = representative.qubit_size

        # Resolve calibration
        cal_id, cal_json, device_cal = self._cal_cache[cal_path]

        # ── E1: Find placements ──
        t_place_start = time.perf_counter()
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
        self._timing["placement_solving_s"] += time.perf_counter() - t_place_start

        # ── Build circuit and observable ──
        t_build_start = time.perf_counter()
        circuit, observable, ham_metadata = self._build_circuit_and_observable(
            ham_name, qsize, representative.topology_edges,
            model_params_override=representative.model_params or None,
        )

        exact_energy = ham_metadata.get("exact_ground_energy")
        self._timing["circuit_build_s"] += time.perf_counter() - t_build_start

        # ── Pre-compute noiseless results (v1.2.0 — two-subprocess pattern) ──
        # Noiseless energy depends on observable + topology but NOT on
        # placement or seed. Compute once per unique topology_hash in a
        # clean subprocess, then pass the cache to the main Pool subprocess.
        # Workers find "noiseless already computed" → skip to noisy envs.
        #
        # Fallback: if precompute fails, workers compute independently
        # (v1.1.1 behavior). Dedup is a performance optimization, not
        # a correctness dependency.
        noiseless_envs = [e for e in tasks[0].noise_configs if e.tier == "noiseless"]
        seen_topologies = set()
        representative_items = []
        if noiseless_envs and placements:
            for placement in placements:
                if placement.topology_hash not in seen_topologies:
                    seen_topologies.add(placement.topology_hash)
                    qubit_names = [
                        placement.qubit_mapping[i] for i in range(qsize)
                    ]
                    representative_items.append((
                        circuit, observable, qubit_names, cal_json, cal_id,
                        "_".join(qubit_names), placement.topology_hash,
                        noiseless_envs,  # only noiseless-tier environments
                        0,  # seed irrelevant for noiseless
                        self._device, {},
                    ))

        noiseless_cache = {}
        t_noiseless_start = time.perf_counter()
        if representative_items:
            print(f"    E2: Pre-computing noiseless for {len(representative_items)} "
                  f"topology group(s) ({len(noiseless_envs)} envs each)")
            noiseless_cache = _precompute_noiseless_subprocess(
                representative_items, self._project_dir,
            )
            if noiseless_cache:
                print(f"    E2: Noiseless cache populated — {len(noiseless_cache)} entries")
            else:
                print(f"    E2: Noiseless cache empty — workers will compute independently")

        self._timing["noiseless_precompute_s"] += time.perf_counter() - t_noiseless_start

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
                    self._device, {},  # cache injected at subprocess level
                ))
                work_meta.append((task, placement, qubit_names, seed))

        # ── Execute batteries ──
        t_exec_start = time.perf_counter()
        cpu_workers = self._config.cpu_workers
        n_items = len(work_items)

        # E2: Route tasks through TieredExecutionPlanner for CPU/GPU decision
        from lumi_hpc_qc.sweep.execution_planner import SimulationTask, select_backend
        exec_backend = select_backend(qsize, method="density_matrix")
        print(f"    E2: Routing {n_items} batteries → {exec_backend} "
              f"({qsize}q, density_matrix)")

        if cpu_workers > 1 and n_items > 1:
            # E2/E7 Level 1: Parallel execution via clean subprocess
            # CRITICAL: The parent process has initialized C++ thread pools
            # (numpy BLAS, h5py HDF5, Aer if imported). Forking directly
            # with mp.Pool causes children to inherit poisoned mutex state,
            # deadlocking at high worker counts (>8). The race condition is
            # scale-dependent: Pool(8) often works, Pool(100) always hangs.
            #
            # Solution: serialize work items to disk, launch a CLEAN
            # subprocess (no inherited C++ state), run Pool there, read
            # results back. Proven in tests/fork_test_subprocess.py:
            # Pool(100) → 100/100 success in 7.43s on LUMI.
            #
            # Cost: ~14KB pickle I/O + subprocess launch (~0.5s).
            # Benefit: guaranteed deadlock-free at any worker count.
            actual_workers = min(self._planner.cpu_workers, n_items)
            print(f"    E2: Parallel execution — {n_items} batteries across "
                  f"{actual_workers} workers (subprocess)")
            t_par = time.time()

            # Close HDF5 before subprocess (not strictly needed since
            # subprocess doesn't inherit, but keeps file clean)
            writer.close()

            battery_results = _run_pool_subprocess(
                work_items, actual_workers, project_dir=self._project_dir,
                noiseless_cache=noiseless_cache,
            )

            # Reopen HDF5 for result writing
            writer.open()
            print(f"    E2: Parallel complete in {time.time()-t_par:.1f}s")
        else:
            # Serial fallback (single worker or single item)
            battery_results = [_battery_worker(item) for item in work_items]

        # ── Process results and write to HDF5 (serial — HDF5 not thread-safe) ──
        self._timing["parallel_execution_s"] += time.perf_counter() - t_exec_start
        t_hdf5_start = time.perf_counter()
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
                    exact_ground_energy=exact_energy,
                    model_params=representative.model_params,
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
        self._timing["hdf5_writes_s"] += time.perf_counter() - t_hdf5_start

    # ── Internal: BYO counts execution (SPEC-002 §7.5 / D3.4) ──

    def _execute_byo_group(
        self,
        tasks: list[SweepTask],
        writer: SweepHDF5Writer,
        errors: list[str],
    ) -> None:
        """Execute a group of byo_circuit tasks (counts -> autocorrelator).

        Unlike the hamiltonian twin battery (_execute_group), the BYO path:
          - builds each task's circuit via _build_byo_circuit (the Gap A seam),
          - solves placements from the BUILT circuit's connectivity
            (extract_connectivity), NOT a topology_library entry,
          - runs each placement x noise env for COUNTS via prepare_simulation
            (device_calibrated -> statevector + per-placement F5a noise;
            noiseless -> statevector), and
          - computes the counts->autocorrelator observable, stored with the
            placement + the noise_placement_independent guardrail flag.

        D3.4a (this step): grouping + dispatch + build + placement solve +
        guardrail resolution. The per-(placement,env) counts run and the
        autocorrelator are stubbed (D3.4b); storage is D3.4c.
        """
        if not tasks:
            return

        representative = tasks[0]
        cal_path = representative.calibration_path
        cal_id, cal_json, device_cal = self._cal_cache[cal_path]

        # ── Build the representative circuit (all tasks in a group share the
        #    circuit_script; connectivity is identical across seeds/grid since
        #    it is the factory's 2q pattern). Use it for the placement solve. ──
        t_build_start = time.perf_counter()
        loaded = self._build_byo_circuit(representative)
        qsize = loaded.num_qubits
        connectivity = loaded.connectivity
        self._timing["circuit_build_s"] += time.perf_counter() - t_build_start
        print(f"    BYO: built {representative.circuit_script} "
              f"({qsize}q, {len(connectivity)} 2q-edges)")

        # ── F5a guardrail (D3a / RED-REVIEW §4 Q2): until per-placement
        #    composition is verified end-to-end, device_calibrated runs on a
        #    SINGLE (top-fidelity) placement; the run is stamped
        #    noise_placement_independent so a placement-blind result is never
        #    read as placement-resolved. noiseless-only groups may use all
        #    placements. ──
        wants_device_cal = any(
            e.source == "device_calibrated"
            for t in tasks for e in t.noise_configs
        )
        max_placements = 1 if wants_device_cal else representative.max_placements
        noise_placement_independent = bool(wants_device_cal)

        # ── Placements from the built circuit's connectivity (top_1 = highest
        #    score, since find_all_placements returns score-descending). ──
        t_place_start = time.perf_counter()
        placements = self._solver.find_all_placements(
            circuit_edges=connectivity,
            circuit_qubits=qsize,
            device_ids=[device_cal.device_id],
            strategy="max_fidelity",
            max_placements=max_placements,
        )
        self._timing["placement_solving_s"] += time.perf_counter() - t_place_start
        if not placements:
            errors.append(
                f"BYO: no valid placements for {representative.circuit_script} "
                f"({qsize}q, edges={connectivity}) on {device_cal.device_id}"
            )
            return
        print(f"    BYO: {len(placements)} placement(s) "
              f"{'(top_1, device_calibrated guardrail)' if wants_device_cal else ''}")

        # ── D3.4b: batched-per-seed counts run -> autocorrelator. Mirrors the
        #    banked floquet_runner_v2 exactly (confirmed by the researcher: one
        #    seed per instance, one run over the whole kick-list), which is what
        #    makes the gate-2 reproduction bit-exact. Within each
        #    (seed, placement, env): build the seed's grid circuits in grid
        #    order, prepare ONE simulation over the list, run with one
        #    seed_simulator = resolve_instance_seed(master_seed, seed), then take
        #    get_autocorrelation per grid point. Average across seeds.
        #    D3.4c stores the per-(placement,env) autocorrelator + counts +
        #    noise_placement_independent (+ physical_qubit_set). For now the
        #    results are computed and returned to a stubbed writer hook. ──
        from lumi_hpc_qc.backends.prepare import prepare_simulation
        from lumi_hpc_qc.sweep.byo_observable import (
            get_autocorrelation, resolve_instance_seed,
        )

        # Source-name mapping: the NoiseConfig name "device_calibrated"
        # (underscore) -> prepare_simulation's VALID_SOURCES "device-calibrated"
        # (hyphen); "noiseless" passes through.
        def _prepare_source(env: NoiseConfig) -> str:
            return "device-calibrated" if env.source == "device_calibrated" else env.name

        # Tasks in this group are (seed x grid-point) for one script+cal. Group
        # by seed; within a seed the grid-point tasks share one disorder
        # instance (identity-shared at expansion, §7.5.4).
        by_seed: dict[int, list[SweepTask]] = {}
        for t in tasks:
            by_seed.setdefault(t.seed, []).append(t)

        # Primary grid axis = the (single) key present in circuit_params. The
        # autocorrelator series must be ordered ascending on it (e.g. num_kicks
        # 0,1,2,...), matching the banked per-kick .dat order. Multi-axis BYO
        # grids are not part of the gate-2 reproduction; guard for it.
        primary_axes = {k for t in tasks for k in t.circuit_params}
        if len(primary_axes) != 1:
            errors.append(
                f"BYO counts path expects a single grid axis (e.g. num_kicks); "
                f"got {sorted(primary_axes)}. Multi-axis BYO counts is a later "
                f"increment (DEBT)."
            )
            return
        primary_axis = next(iter(primary_axes))

        # The set of distinct noise envs (same across tasks in a group).
        envs = representative.noise_configs
        # init_bit_array / num_qubits for the observable (from disorder + fixed).
        init_bit_array = representative.disorder_instance.get("init_bit_array")
        if init_bit_array is None:
            errors.append(
                f"BYO: disorder instance has no init_bit_array "
                f"(needed for the autocorrelator) — {representative.circuit_script}"
            )
            return

        t_exec_start = time.perf_counter()
        byo_results: list[dict] = []  # one per (seed, placement, env)
        for placement in placements:
            phys_qubits = [placement.qubit_mapping[i] for i in range(qsize)]
            phys_edges = [
                (phys_qubits[a], phys_qubits[b]) for (a, b) in connectivity
            ]
            for seed, seed_tasks in sorted(by_seed.items()):
                # Build the seed's grid circuits in grid order (primary axis asc).
                seed_tasks_sorted = sorted(
                    seed_tasks, key=lambda tk: tk.circuit_params[primary_axis],
                )
                built = [self._build_byo_circuit(tk).circuit for tk in seed_tasks_sorted]
                inst_seed = resolve_instance_seed(representative.master_seed, seed)

                for env in envs:
                    src = _prepare_source(env)
                    prep_kwargs = dict(
                        calibration_path=cal_path, num_qubits=qsize,
                        optimization_level=3, num_processes=1, verbose=False,
                    )
                    if src == "device-calibrated":
                        prep_kwargs.update(
                            physical_qubits=phys_qubits, physical_edges=phys_edges,
                        )
                    prep = prepare_simulation(built, src, **prep_kwargs)
                    job = prep.simulator.run(
                        prep.run_circuits, shots=env.shots, memory=True,
                        seed_simulator=inst_seed,
                    )
                    result = job.result()
                    autocorr = [
                        get_autocorrelation(result.get_counts(i), init_bit_array, qsize)
                        for i in range(len(built))
                    ]
                    byo_results.append({
                        "seed": seed,
                        "placement_id": placement.placement_id,
                        "physical_qubit_set": phys_qubits,
                        "env": env.name,
                        "noise_source": env.source,
                        "noise_placement_independent": noise_placement_independent,
                        "num_kicks": [
                            tk.circuit_params[primary_axis]
                            for tk in seed_tasks_sorted
                        ],
                        "autocorrelator": autocorr,
                        "shots": env.shots,
                        "seed_simulator": inst_seed,
                    })
        self._timing.setdefault("byo_exec_s", 0.0)
        self._timing["byo_exec_s"] += time.perf_counter() - t_exec_start
        print(f"    BYO: computed {len(byo_results)} (seed x placement x env) "
              f"autocorrelator series")

        # ── D3.4c: persist byo_results via the writer (HDF5 + Parquet, with the
        #    noise_placement_independent flag + physical_qubit_set). Stubbed
        #    until D3.4c so D3.4b is verifiable on the compute path alone. ──
        self._byo_results_last = byo_results  # surfaced for D3.4b verification
        return

    # ── Internal: circuit building ──

    def _build_circuit_and_observable(
        self,
        hamiltonian_name: str,
        num_qubits: int,
        edges: list[tuple[int, int]],
        model_params_override: dict[str, float] | None = None,
    ) -> tuple[Any, Any, dict[str, Any]]:
        """Build a reference circuit and observable for evaluation.

        For characterization sweeps, builds a simple measurement
        circuit from the hamiltonian via the plugin registry.

        The sweep YAML uses domain-language names ("tfim", "heisenberg",
        "fermi_hubbard") which map directly to plugin registry names.
        Shorthand aliases ("fh" → "fermi_hubbard") are resolved here.

        Args:
            hamiltonian_name: Plugin name or alias.
            num_qubits: Number of qubits.
            edges: Topology edges.
            model_params_override: Optional dict of parameter overrides
                from LHS sampling (v1.2.0 Item C). Merged over defaults.

        Returns:
            (circuit, observable, metadata_dict)
        """
        from qiskit import QuantumCircuit
        from lumi_hpc_qc.types import ExperimentConfig

        # Resolve aliases
        plugin_name = _SWEEP_TO_PLUGIN.get(hamiltonian_name, hamiltonian_name)

        # Build default model_params via the plugin's own default_params()
        # (v1.2.1 Item 3 — replaces centralized switch statement)
        ham_builder = self._registry.get_hamiltonian(plugin_name)
        model_params = ham_builder.default_params(num_qubits)
        if model_params_override:
            model_params.update(model_params_override)

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

    @staticmethod
    def _build_byo_circuit(task: SweepTask):
        """Per-task BYO build seam (SPEC-002 §7.5).

        Assembles the factory kwargs as fixed ∪ disorder ∪ grid-point and builds
        the concrete circuit for this task. The same disorder object was attached
        to every grid point in the task's seed at expansion time, so the
        cross-grid invariant already holds (verified once per experiment in
        _expand_byo_experiment); this method just realizes one point.

        Returns the LoadedCircuit (circuit + extracted connectivity), which the
        BYO execution path uses for placement (connectivity, not a topology
        library entry) and counts-based evaluation. Not yet invoked by
        _execute_group — the BYO execution path lands with Gap B/D3.
        """
        from lumi_hpc_qc.sweep.byo_sweep import assemble_build_kwargs
        from lumi_hpc_qc.sweep.circuit_loader import load_circuit

        build_kwargs = assemble_build_kwargs(
            task.fixed_params, task.disorder_instance, task.circuit_params,
        )
        return load_circuit(
            script_file=task.circuit_script,
            script_function=task.circuit_function,
            script_params=build_kwargs,
        )


# ═══════════════════════════════════════════════════════════════════════
# Plugin name resolution
# ═══════════════════════════════════════════════════════════════════════

# Map sweep YAML shorthand aliases → plugin registry names
_SWEEP_TO_PLUGIN: dict[str, str] = {
    "fh": "fermi_hubbard",
    "qaoa": "qaoa_maxcut",
}



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
