<!-- Copyright (c) 2026 Michael Mucciardi -->
<!-- SPDX-License-Identifier: SSPL-1.0 -->

# Changelog

## 1.4.0 (2026-04-10)

### Global Pool Packing, Seed Lists, Parquet 67→71 (RED-RESP-V140-DESIGN-v1.0 REVISED)

Cross-experiment QPU packing for maximum throughput.  ~608 new source lines
across 6 files + ~429 test lines.  All 7 acceptance tests pass.  13/13 SLURM
regression suites passing (LUMI jobs 17406427–17406439).

**Governing documents:** RED-RESP-V140-DESIGN-v1.0 (REVISED),
ORANGE-TO-RED-COMMS-023, BLUE-DESIGN-V140-v1_0

#### Item 1 — GlobalPoolPacker (`sweep/mixed_packing.py`, ~250 lines)
- `PoolTask` dataclass: atomic packing unit (circuit + qubits + edges + metadata)
- `PackedBatch` dataclass: one QPU submission (tasks + utilization)
- `GlobalPoolPacker`: first-fit-decreasing bin packing across all experiment groups
  - O(N × B) greedy backfill, deterministic with packing seed
  - Constraint: no qubit overlap AND no CZ edge overlap
  - `objective` parameter: `max_throughput` implemented, `capped_utilization` and
    `single_topology` accepted but raise ValueError (deferred to v1.4.1+)
- `validate_packed_batch()`: per-batch 3-invariant check (qubits, edges, task IDs)
- `validate_packing()`: campaign-level validation (every task exactly once)
- Device qubits derived from `DeviceCalibration.num_qubits` (no hardcoded constants)

#### Item 2 — Pre-build Helper (`backends/pauli_measurement.py`, ~77 lines)
- `prebuild_pool_tasks()`: builds PoolTasks from (seed, placement, observable)
- Corrected signature per RED-RESP-V140 §4a: `device_cal`, `hamiltonian_name`,
  `topology_name` parameters added alongside existing circuit/observable/shots
- Edge tuples derived from `device_cal.adjacency` (not `Placement.internal_edges`
  which is an int count, not a set)
- 7 metadata fields per task: seed, placement_id, pauli_group_index,
  pauli_group_labels, identity_energy, hamiltonian, topology_name

#### Item 3 — Packing Manifest (`sweep/mixed_packing.py`, ~80 lines)
- `PackingManifest`: static record of task→batch assignment
- Written atomically (temp file + rename) after `pack()`, before QPU submission
- Schema per ORANGE-TO-RED-COMMS-023 §4: strategy, objective, packing_seed,
  device_qubits, per-batch task list with full provenance
- `save()` / `load()` round-trip with crash-safe atomic writes
- `completed_batch_ids()`: cross-reference with CampaignManifest for resume
- Resume flow: replay original packing + skip completed batches

#### Item 4 — `seed_list` Support (`sweep/sweep_engine.py`, ~40 lines)
- `SweepExperimentConfig.seed_list: list[int] | None` field
- Three YAML formats: explicit list `[0, 5, 42]`, range string `"0-4,10-14,42"`,
  single integer `42`
- `_parse_seed_range()`: comma-separated tokens, dash ranges inclusive both ends
- When `seed_list` present, `seeds`/`seed_offset` are ignored
- Validation: empty list, negative values, duplicates all caught
- Both LHS and standard grid expansion paths use resolved `seed_values`

#### Item 5 — Packing Config (`sweep/sweep_engine.py`, ~30 lines)
- `PackingConfig` dataclass: strategy (`dsatur`|`global_pool`), objective, seed
- Top-level in `SweepConfig` (not per-experiment — RED-RESP-V140 §2 Q1)
- `parse_sweep_config()` reads `sweep.packing` YAML section
- `validate_sweep_config()` checks strategy against known values

#### Item 6 — Science Parquet 67→71 (`data/sweep_export.py` + `data/hdf5_writer.py`)
- Column 68: `calibration_set_id` (string, nullable) — VTT QX calibration UUID
- Column 69: `packing_co_placements` (int32) — tasks in batch
- Column 70: `packing_qubit_utilization` (float64) — batch utilization
- Column 71: `packing_algorithm` (string) — `"dsatur"` | `"global_pool"` | `"none"`
- Defaults for unpacked runs: 1, 0.0, `"none"`
- `SweepResultEntry`: 4 new fields with defaults
- HDF5 attrs: all 4 persisted, WAL round-trip (serialize + deserialize)
- Section comments corrected: Device & Placement (7→10), Calibration (10→11)

#### Item 7 — Public API Exports (`sweep/__init__.py`)
- Exports: PoolTask, PackedBatch, GlobalPoolPacker, PackingManifest,
  validate_packed_batch, validate_packing, PackingConfig
- Docstring updated: 67→71 columns, added mixed_packing + campaign_manifest

### Validation
- E6b: 10 new test sections (E6b.11–E6b.20)
  - AT1: No qubit overlap in any packed batch
  - AT2: No CZ edge overlap in any packed batch
  - AT3: Every task from pool appears exactly once in packing
  - AT4: Deterministic — same pool + same seed = identical assignment
  - AT5: PackingManifest save/load round-trip + provenance preservation
  - AT6: Mixed-topology batches (4q chain + 2q Bell in one composite)
  - AT7: Objective validation (unknown raises ValueError, unimplemented raises ValueError)
  - E6b.19: seed_list parsing, expand_grid, validation, PackingConfig
  - E6b.20: 71-column schema verification (count, names, insertion order)
- E8: 67→71 column count, 4 new column names in expected list
- E10: 67→71 column count in end-to-end pipeline
- SLURM e6b time limit: 15s → 30s (doubled test count)
- 13/13 regression: LUMI SLURM jobs 17406427–17406439, all PASS

## 1.3.1 (2026-04-10)

### QPU Behavior Audit — Configurable Defaults (RED-DIRECTIVE-QPU-CONFIG-v1.0)

Mandatory patch before any production QPU campaign.  All automatic QPU
behaviors are now off by default and configurable via the new `qpu:`
YAML section.  ~130 lines modified across 4 source files + 2 test files.

**Governing document:** RED-DIRECTIVE-QPU-CONFIG-v1.0

#### Finding 1 — Retry Disabled by Default (CRITICAL)
- Removed hardcoded `MAX_RETRIES = 3` and `RETRY_BASE_WAIT_S = 1`
- Retry is OFF by default — errors propagate immediately
- When enabled via `qpu.retry.enabled: true`, only errors matching
  `retryable_errors` patterns trigger retry; fatal errors always propagate
- `except Exception` no longer silently retries permanent failures

#### Finding 2 — QPUConfig Dataclass + YAML `qpu:` Section
- New `QPUConfig` dataclass in `sweep_engine.py` (10 fields, all safe defaults)
- `parse_sweep_config()` parses `qpu:` section from campaign YAML
- `IqmQpuBackend.set_qpu_config()` for post-construction application
- All QPU params flow from YAML → QPUConfig → backend (single source)

#### Finding 3 — Auto-Chunk Logging
- Clear warning when batch exceeds VTT limit and is split into chunks
- Shows chunk count, sizes, and queue wait impact

#### Finding 4 — QXClient / Timing Capture Opt-In
- `QXClient.from_backend()` only created when `qpu.timing_capture: true`
- Queue length prefetch only when `qpu.queue_prefetch: true`
- `get_job_policy()` only when `batch_limit` not set in config
- Zero automatic HTTP requests to VTT QX API by default

#### Finding 5 — Connection Timeout
- `signal.alarm()` wraps `IQMProvider.get_backend()` (default 60s)
- Clear error message with calibration schedule and config hint
- Configurable via `qpu.connection_timeout_s`

#### Finding 6 — Shots Single Source
- `IqmQpuBackend._shots` reads from `QPUConfig.shots`
- Hardcoded 4096 defaults remain in 5 modules as fallbacks only;
  QPUConfig is the authoritative source when present

#### Finding 7 — retry_attempts in Benchmark Parquet
- Schema updated: 35 → 36 columns (+`retry_attempts` int32 nullable)
- `IqmQpuBackend.get_batch_retry_attempts()` parallel to `get_batch_timings()`
- Sweep engine injects retry counts into timing records before export
- Null for simulator sweeps

### Validation
- E6b: updated retry tests → QPUConfig defaults, `set_qpu_config()`,
  `get_batch_retry_attempts()`
- E7: updated Parquet column count 35 → 36, added `retry_attempts` column check

## 1.3.0 (2026-04-09)

### VTT QX API Integration, Campaign Reliability, Benchmark Parquet (RED-DIRECTIVE-V130-v1.0)

8 items + 2 bug fixes + version alignment. 712 new/modified lines across 9 files.
All code validated live on Q50 (April 7–8). 13/13 SLURM regression suites passing
(809 checks across E1–E10, V111). Branch `v1.3.0-qx-integration` merged to main.

**Governing documents:** RED-DIRECTIVE-V130-v1.0, RED-DIRECTIVE-BENCHMARK-PARQUET-v1.0,
RED-RESP-ORANGE-COMMS-019-v2.0, ORANGE-TO-RED-COMMS-019-v2.1

#### Item 1 — Benchmark Parquet Export (`data/benchmark_export.py` NEW, 266 lines)
- 35-column PyArrow schema per RED-DIRECTIVE-BENCHMARK-PARQUET-v1.0 Appendix A
- `export_benchmark_to_parquet()`: one row per QPU batch, server-side timing from
  QX API timeline, packing metrics from DSatur round data
- `make_simulator_timing_records()`: converts `sweep_timing.json` for simulator sweeps
- QPU-specific columns (server timing, context) nullable — null for simulator mode
- `sweep_benchmark.parquet` written alongside `sweep.h5` at sweep completion

#### Item 2 — QXClient Wired into iqm_qpu.py (`backends/iqm_qpu.py`)
- `QXClient.from_backend()` created in `_ensure_sim()` after QPU connection
- `capture_job_timing()` replaces raw `.run()` in `_submit_batch()`
- `QPUJobTiming` records accumulated in `_batch_timings` — one per batch
- `get_batch_timings()` accessor for benchmark Parquet pipeline
- Queue length fetched once before sweep via `get_queue_length()`

#### Item 3 — Dynamic Batch Limit (`backends/iqm_qpu.py`)
- `get_job_policy()` queried at connection time in `_ensure_sim()`
- `VTT_BATCH_LIMIT` overridden from `max_number_circuits_per_batch`
- Fallback: 100 if endpoint unreachable. Source logged.

#### Item 4 — Version Alignment (MERGE GATE)
- `pyproject.toml`: `"1.1.0"` → `"1.3.0"`
- `__init__.py`: `"1.2.4"` → `"1.3.0"`

#### Item 5 — Campaign Manifest (`sweep/campaign_manifest.py` NEW, 225 lines)
- `CampaignManifest`: tracks task completion across QPU batch submissions
- Atomic persistence: temp file + `os.rename()` — crash-safe
- `pending_tasks()` returns PENDING + FAILED tasks for resume
- `campaign_manifest.json` alongside `sweep.h5` in output directory

#### Item 6 — Sweep Resume (`sweep/sweep_engine.py`)
- Load existing manifest at sweep start, filter completed tasks
- After each group: mark completed/failed + atomic save
- Resume is idempotent

#### Item 7 — Batch Retry with Exponential Backoff (`backends/iqm_qpu.py`)
- 3 retries, 1s/2s/4s backoff around `_submit_batch()` calls
- Transient failures retried; permanent failures propagate

#### Item 8 — Edge-Overlap Fix (`sweep/mixed_packing.py`, MANDATORY)
- `device_cal` required in `MixedPacker.__init__()` — `ValueError` if None
- Prevents packing placements sharing CZ edge without qubit overlap

#### Bug Fix A — E6a Demux Uniform Qubit Count (`sweep/demultiplexer.py`)
- `num_logical` moved inside per-placement loop (lines 39, 116)
- Was taking `placements[0]` length — wrong for mixed qubit counts

#### Bug Fix B — Edge Direction Normalization (`tests/fetch_q50_calibration.py`)
- `_normalize_edge()`: sorts pair to canonical form (`QB3-QB4`)
- Eliminated ~40 false warnings in topology completeness check

### Validation
- 13/13 SLURM suites: 809 checks (784 existing + 25 new)
- E6a.7: Mixed qubit count demux (5 checks)
- E6b.10: device_cal guard + retry constants (6 checks)
- E7.13: Campaign manifest + benchmark Parquet (14 checks)

## 1.2.4 (2026-04-07)

### CRITICAL BUG FIX — Shot-Based Energy Computation (RED-FINDING-EVAL-RUNNER-v1.0)

**Bug:** `_energy_from_counts()` in `eval_runner.py` treated X Pauli operators as
identity, contributing `coefficient × (+1)` instead of `0.0` for Z-basis
measurements. All shot-based `best_energy` values for Hamiltonians with X or Y
terms were incorrect (e.g., TFIM 4q |0000⟩: -7.0 instead of -3.0).

**Root cause:** Three copies of the same Z-only parity calculation existed in the
codebase. The correct `pauli_measurement.py` module (263 lines, basis rotation)
was wired into `aer_gpu.py` (VQE path) but never into `eval_runner.py` (sweep
characterization path), `demultiplexer.py` (QPU demux), or `mixed_packing.py`.

**Fix (`sweep/eval_runner.py`)**
- Shot-based branch rewritten: uses `build_measurement_circuits()` to create
  one circuit per qubit-wise-commuting Pauli group with proper basis rotation
  (H gate for X, S†H for Y), transpiles all circuits together, runs as batch,
  combines via `expectation_from_grouped_counts()` — matches `aer_gpu.py` pattern
- Circuits built from untranspiled source so transpiler routes both original
  gates and rotation gates together

**Deprecation guards (3 files)**
- `eval_runner.py::_energy_from_counts()` — raises `ValueError` for non-Z/I terms
- `demultiplexer.py::_energy_from_counts()` — raises `ValueError` for non-Z/I terms
- `mixed_packing.py::compute_mixed_energies()` — raises `ValueError` for non-Z/I terms
- Parity check tightened from `("Z", "Y")` to `("Z",)` as defense-in-depth

**Hardened (`backends/pauli_measurement.py`)**
- `expectation_from_counts_direct()` warning upgraded to `ValueError` for non-Z terms

**Validation (`tests/e5_byo_eval_validation.py`)**
- E5.8: 8-check cross-path validation suite
  - TFIM 4q exact (statevector) = -3.0 ✓
  - TFIM 4q shot-based (4096 shots) matches exact within 0.3
  - Shot-based energy is NOT the buggy -7.0
  - `_energy_from_counts()` raises ValueError for TFIM (has X terms)
  - `_energy_from_counts()` still works for pure-Z observable
  - Bell state cross-path validation (non-trivial state)

**Validation (`tests/e7_sweep_validation.py`)**
- E7.9b: 2-check F1 regression in sweep output
  - No shot energy exceeds exact_ground_energy by >2.0
  - No energy >3.0 below exact ground state (F1 bug signature)

**Affected data:** All CAMPAIGN-001 rows (12,175). Noiseless atlas (shots=0) and
VQE campaigns (aer_gpu.py path) unaffected.

**Found by:** Team Orange (COMMS-017-v1.1), confirmed by Team Red
(RED-FINDING-EVAL-RUNNER-v1.0). Two additional instances found by Team Blue.

## 1.2.3 (2026-04-06)

### Sweep Timing JSON (RED-DIRECTIVE-V123)

Single item. Lightweight timing harness captures per-phase wall time for
every sweep, written as `sweep_timing.json` alongside `sweep.h5`.
Establishes baseline for regression detection and cost modeling.

**Sweep Timing Harness (`sweep/sweep_engine.py`)**
- `time.perf_counter()` marks at every phase boundary in `run()` and `_execute_group()`
- Per-group accumulation: placement_solving, circuit_build, noiseless_precompute,
  parallel_execution, hdf5_writes — summed across groups
- Top-level phases: config_parse, calibration_load, grid_expansion
- `_write_timing_json()` writes structured JSON at sweep completion
- `_get_stripe_info()` captures Lustre stripe count/size via `lfs getstripe`
  (graceful fallback when lfs unavailable)
- Sampling config block present for LHS sweeps, absent for grid mode
- Storage context block present on Lustre, absent elsewhere
- Environment metadata: SLURM node, partition, CPUs, job ID, container
- Timing never fails a sweep — JSON write errors are silently caught

**E7 Validation (`tests/e7_sweep_validation.py`)**
- E7.12: 10 new checks — JSON exists, parseable, total_elapsed > 0,
  all phases non-negative, sum(phases) ≤ total, workers > 0, tasks > 0,
  environment node present

## 1.2.2 (2026-04-06)

### HPCQC_PLUGIN_PATH for Read-Only Containers (RED-DIRECTIVE-V122)

Single-item patch. Adds filesystem-based plugin discovery for HPC environments
where containers are read-only and `pip install` is not available.

**HPCQC_PLUGIN_PATH (`plugins/registry.py`)**
- Third discovery phase: `_discover_plugin_path()` after built-in and entry point scans
- Reads `HPCQC_PLUGIN_PATH` environment variable (colon-separated directory list)
- Scans subdirectories matching `_PLUGIN_TYPES` keys (e.g., `hamiltonians/`, `ansatze/`)
- Loads loose `.py` files via `importlib.util.spec_from_file_location()` — no package
  structure or `__init__.py` required
- P1: Same ABC validation as entry points
- P2: Priority order: built-in > entry points > plugin path (never overrides)
- P3: Audit logging with file path for provenance
- P4: Files starting with `_` skipped (convention for non-plugin helpers)
- Enables Orange's plugin deployment on LUMI without `pip install` or file copying:
  `export HPCQC_PLUGIN_PATH="/path/to/animll/plugins"`

## 1.2.1 (2026-04-06)

### Plugin Architecture Completion (RED-DIRECTIVE-V121 v1.1)

6 items. Plugin system extended: external packages contribute plugins via
entry points, plugins provide their own defaults, calibration routing is
explicit. No QPU dependency.

**Item 1: Calibration Adapter Registry Integration (`plugins/registry.py`, `sweep/sweep_engine.py`)**
- `calibration_adapters` added as 7th plugin type in `_PLUGIN_TYPES`
- `get_calibration_adapter(name)` typed accessor on `PluginRegistry`
- `_discover_builtin()` now scans calibration_adapters directory alongside all other plugin types
- `registry.list_available("calibration_adapters")` returns `["iqm_v2", "synthetic"]`

**Item 2: Entry Point Plugin Discovery (`plugins/registry.py`)**
- `_discover_entrypoints()` scans `hpcqc.plugins.*` entry point groups from pip-installed packages
- 7 groups: hamiltonians, ansatze, optimizers, gradients, initializers, error_mitigation,
  calibration_adapters — derived from `_PLUGIN_TYPES` keys
- R1: ABC validation — entry point must subclass the correct ABC or is skipped with warning
- R2: Built-in priority — built-in plugins are never overridden by entry points
- R3: Audit logging — source package name + version printed for every external plugin loaded
- R4: Failure isolation — broken entry points produce warnings, not crashes
- Enables Orange's DiagnosticTFIM deployment via `pip install -e animll` instead of file copying

**Item 3: `default_params()` on HamiltonianBuilder ABC (`plugins/hamiltonians/base.py`)**
- Non-abstract `default_params(num_qubits)` method returns plugin-specific defaults
- Base implementation returns `{"num_qubits": num_qubits}` — all existing plugins work unchanged
- `tfim.py`: returns `{num_qubits, j=1.0, g=1.0, boundary_condition="open"}`
- `heisenberg.py`: returns `{lattice_rows=1, lattice_cols, jx=1.0, jy=1.0, jz=1.0}`
- `fermi_hubbard.py`: returns `{lattice_rows=1, lattice_cols, hopping_t=1.0, interaction_u=2.0}`
- Eliminates centralized `_default_model_params()` switch statement in `sweep_engine.py`
- External plugins (DiagnosticTFIM) override to provide correct param columns in Parquet

**Item 4: Explicit `adapter` Field in Calibration JSON (`sweep/sweep_engine.py`)**
- `_load_calibration()` reads `cal_json.get("adapter", "iqm_v2")` — explicit routing
- Replaces `_detect_adapter()` heuristic (removed)
- Default `"iqm_v2"` for backward compat with files lacking the field
- Single JSON read (was two: one for detection, one for noise model)
- New devices add `"adapter": "aalto_q20"` to their calibration JSON

**Item 5: Entry Point Group Names in API Stability (documentation)**
- 7 entry point group names (`hpcqc.plugins.*`) committed as stable
- Will not be renamed without a deprecation cycle

**Item 6: Dual-Name Parameter Extraction Fix (`data/sweep_export.py`)**
- `param_j` checks both `model_params["j"]` and `model_params["coupling_j"]`
- `param_g` checks both `model_params["g"]` and `model_params["transverse_h"]`
- `param_disorder_w` checks both `model_params["disorder_w"]` and `model_params["w"]`
- Fixes null Parquet columns when LHS YAML uses long parameter name aliases

**Performance: Registry cached on SweepEngine (`sweep/sweep_engine.py`)**
- `PluginRegistry` instantiated once in `__init__`, shared across all methods
- Was: 2 separate instantiations + `discover()` calls per sweep
- Eliminates redundant plugin directory scans during execution

**SLURM Wall Times Tightened**
- All test scripts set to ~2× observed runtime (was 15–60 min, now 15s–4min)
- Fail-fast on hangs: runaway forks killed in seconds, not minutes

## 1.2.0 (2026-04-05)

### Noiseless Dedup Restoration, LHS Sampling, Exact Ground Energy (RED-SPEC-003)

Items A, C, D from RED-SPEC-003. 117/117 validation checks on LUMI (55s).
Parquet schema: 67 columns (was 63). Item B (cross-seed pool dispatcher) deferred to v1.2.1.

**Item A: Noiseless Dedup Restoration — Two-Subprocess Pattern (`sweep/sweep_engine.py`)**
- `_precompute_noiseless_subprocess()` runs noiseless-tier simulations (noiseless +
  topology_noiseless) once per unique topology group in a clean subprocess
- Cache dict passed to main Pool subprocess via separate pickle file
- Workers find noiseless results pre-computed → skip 2 envs, run only 9 noisy envs
- VE18: 1800 simulations + 400 deduplicated = 2200 HDF5 writes (was 2200 + 0 in v1.1.1)
- Graceful fallback: if precompute subprocess fails, workers compute independently
- RED-SPEC-003 count bound assertion: `deduplicated >= (placements - topology_groups) × 2`

**Item C: LHS Sampling (`sweep/sweep_engine.py`, `data/hdf5_writer.py`, `data/sweep_export.py`)**
- `SamplingConfig` dataclass: method ("grid"|"lhs"), n_samples, parameter ranges, seed
- `_generate_lhs_samples()` via `scipy.stats.qmc.LatinHypercube` — quasi-random sampling
  with uniform marginal coverage across all parameter dimensions
- `model_params: dict[str, float]` on `SweepTask` — per-task Hamiltonian parameter overrides
- LHS samples merged over plugin defaults in `_build_circuit_and_observable()`
- Task grouping key extended to 4-tuple: (hamiltonian, topology, calibration, params_hash)
- HDF5 group path includes `/params_{hash}` suffix when model_params non-empty to prevent
  path collision across LHS samples sharing the same placement/seed/calibration
- `model_params` persisted as JSON attribute on HDF5 result groups
- Parquet: 3 typed float64 columns per RED-SPEC-003 (Red rejected JSON column):
  `param_j`, `param_g`, `param_disorder_w` — null when parameter not set
- Hamiltonian-agnostic: any plugin reads from `config.model_params`; YAML parameter
  names match plugin keys (TFIM: j/g, Heisenberg: jx/jy/jz, etc.)
- YAML interface: `sampling: {method: lhs, n_samples: 1000, parameters: {j: [0.5, 2.0], g: [0.5, 2.0]}, seed: 42}`

**Item D: Exact Ground Energy Persistence (`data/hdf5_writer.py`, `data/sweep_export.py`)**
- `exact_ground_energy` computed via `eigvalsh(H.to_matrix())` for ≤24 qubits, now persisted
  as HDF5 float64 attribute on each result group (was computed then discarded in v1.1.1)
- Parquet: `exact_ground_energy` column in Hamiltonian Properties section — null for >24 qubits
- Export reads from HDF5 attribute (self-contained); falls back to external `exact_energies`
  dict for backward compatibility with v1.1.x HDF5 files
- `exact_energy` and `relative_error` Parquet columns now populated without caller input

**Schema Evolution: 63 → 67 Columns**
- `exact_ground_energy` (float64, nullable) — Hamiltonian Properties
- `param_j` (float64, nullable) — Model Parameters
- `param_g` (float64, nullable) — Model Parameters
- `param_disorder_w` (float64, nullable) — Model Parameters

## 1.1.1 (2026-04-05)

### DSatur Packing, IQM Batch, Multiprocessing Deadlock Fix (RED-DIRECTIVE-V111)

- DSatur graph coloring for QPU multi-placement packing (circuit_composer.py)
- IQM batch submission chunking: 200-circuit limit with auto-splitting
- mp.Pool fork deadlock fixed: Pool runs in clean subprocess (subprocess.Popen
  with pickle serialization, not direct mp.Pool in main process)
- Parquet schema: 63 columns (was 61) — packing_round, packing_group_size added

## 1.1.0 (2026-04-03)

### Phase E — General Sweep Engine (RED-SPEC-002 + RED-DIRECTIVE-E4-SCHEMA)

Complete sweep engine for systematic noise characterization across QPU topologies.
764 checks, zero failures. 22/22 VE criteria satisfied. 11 E-steps.

**E1: Placement Solver** — General connected subgraph enumeration for any device topology.
Calibration adapter interface for IQM v2 JSON format.

**E2: Execution Planner** — Tiered execution: Tier 0 (noiseless, statevector, CPU),
Tier A (noisy simulation, density matrix, CPU), Tier 1 (QPU). CPU parallelism validated
at 128-way on LUMI.

**E3: HDF5 Writer** — Hierarchical storage: device/placement/seed/noise_config.
SWMR mode for crash recovery. WAL-based recovery via `recover_from_wal()`.

**E4: Twin Simulator** — 11 noise environments per placement: noiseless, topology_noiseless,
9 decomposed noise channels. Each environment isolates one noise mechanism.

**E5: BYO Circuits + Eval-Only** — Circuit loader (QPY/QASM/script), evaluation-only
mode for non-parameterized circuits.

**E6a: Multi-Round Packing** — Composite circuits pack multiple placements onto one device.
DSatur coloring, round-robin executor.

**E6b: Mixed-Experiment Packing** — Different circuit types in the same composite circuit.
Per-entry observable tracking and demultiplexing.

**E7: Sweep Engine Orchestrator** — YAML-driven, multiprocessing Pool in subprocess,
config expansion, HDF5 accumulation, Parquet/HDF5 export.

**E8: Sweep Export** — 61-column Parquet schema. HDF5 → Parquet/CSV pipeline.

**E9: Synthetic Calibration CLI** — Gaussian perturbation of real calibration data
for sensitivity studies. CLI: `python -m lumi_hpc_qc.sweep.perturb_calibration`.

**E10: Validation + FiQCI Examples** — End-to-end validation, FiQCI circuit builder.

## 1.0.0 (2026-04-01)

### Framework Baseline — 5-Point Benchmark (RED-SPEC-001 complete)

All V1–V20 satisfied. 312+ validated checks across Phases A–D. 229/229 at tag time.

**Phases A–D implemented:**
- A: Unit tests, launcher, ham_meta, ansatz validation
- B: Parameterized noise, topology-noiseless, circuit metrics, config gen, placement solver
- C: Readout mitigation, ZNE (mitiq), multi-seed, multiplexed QPU, reproducibility
- D: Multi-format export (Parquet/HDF5/JSONL/NPZ/CSV), schema v2, quality gates

**5-point benchmark (SLURM validated):**

| Model | Noiseless | Controlled | Noisy | QPU |
|---|---|---|---|---|
| TFIM 2q | ✓ | ✓ | ✓ | ✓ |
| TFIM 4q | ✓ | ✓ | ✓ | ✓ |
| TFIM 8q | ✓ | ✓ | ✓ | ✓ |
| H₂ 4q | ✓ | ✓ | ✓ | ✓ |
| QAOA 8q | ✓ | ✓ | ✓ | ✓ |

## 1.0.0b7 (2026-03-31)

### Phase D — Multi-Format Export, Schema v2, Quality Gates (RED-SPEC-001)
- Multi-format export pipeline: Parquet, HDF5, JSONL, NPZ, CSV (enriched)
- QPY export script (`scripts/qpy_export.py`) for circuit serialization
- Schema v2.0.0 (`data/schema.py`) with jsonschema validation
- Data quality gate (`data/quality.py`) — 5 pre-write checks, never blocks writes
- ExperimentRecord metadata enrichment: spectral_gap, hamiltonian_locality,
  noiseless_tier, error_mitigation_applied, per_placement_results
- v1→v2 record upgrade path for backward compatibility
- CLI: `python -m lumi_hpc_qc.data.export results/*.json --format all`
- Cross-implementation validation script (`scripts/cross_impl_validation.py`)
- **86/86 SLURM-verified checks** (jobs 17116765, 17117032)

## 1.0.0b6 (2026-03-30)

### Phase C — Error Mitigation, Multiplexed QPU, Statistical Characterization (RED-SPEC-001)
- Readout error mitigation (`plugins/error_mitigation/readout.py`) — tensor product
  calibration matrix inversion, integrated into VQA workflow
- ZNE via mitiq (`plugins/error_mitigation/zne_mitiq.py`) — lazy import to avoid
  MPI_Init_thread conflict, linear/polynomial/exponential extrapolation
- Multi-seed sweep (`scripts/generate_seed_sweep.py`) — unique-but-reproducible seeds
- Multiplexed QPU circuit construction (`backends/circuit_packing.py`) — pack multiple
  placements into device-width circuit, 12× throughput improvement
- Result demultiplexer — extract per-placement counts from composite results
- Measurement statistics (V19) — bitstring entropy, parity, effective rank
- **48/48 SLURM-verified checks** (job 17095335)

## 1.0.0b5 (2026-03-30)

### Phase B — Parameterized Noise, Topology-Noiseless, Circuit Metrics (RED-SPEC-001)
- Parameterized noise model with per-channel isolation
- Topology-noiseless mode: real coupling map, ideal gates
- Circuit metrics: depth, gate count, CNOT count, connectivity degree
- Config generator: systematic noise decomposition across 91 configs
- Connectivity-aware placement solver: greedy connected subgraph
- **37/37 SLURM-verified checks** (job 17094955)

## 1.0.0b4 (2026-03-30)

### Phase A — Unit Tests, Launcher, Ansatz Validation (RED-SPEC-001)
- Unit test suite: config_loader (12), hamiltonians (5), pauli_measurement (6), provenance (8)
- Launcher fixes: env.sh sourcing, PYTHONHASHSEED, SLURM guard
- Hamiltonian metadata: spectral gap, locality, pauli count
- Ansatz validation: 13 lines across 3 plugins, prevents cryptic crashes
- TFIM reference correction: −(1+√2) → −√5 for 2q
- HVA 2-qubit crash fix
- **31/31 SLURM-verified checks** (job 17093900)

## 1.0.0b3 (2026-03-28)

### Critical Bug Fixes + Design Concern Resolutions

**F1 CRITICAL: Pauli measurement bug fixed**
- `_expectation_from_counts()` silently dropped X and Y Pauli terms
- Only Z-basis parity was computed; X/Y contributions returned +1 instead of 0
- Fix: `pauli_measurement.py` module with proper basis rotation (H for X, Sdg+H for Y)
- Wired into VQE path (`aer_gpu.py`)

**F2 CRITICAL: Backend bypass fixed**
- VQE `eval_energy()` called `self._sim.run()` directly — bypassed backend interface
- Fix: all evaluations go through `Backend.run_circuits()` abstract method

**C1–C4: Noise model fixes**
- C1: Noise model now topology-aware (coupling map constraint)
- C2: Benchmark configs decomposed — one variable at a time
- C3: H₂ noiseless config uses validated optimizer stack
- C4: Double-counted noise on single-qubit gates removed
  (depolarizing from RB data already includes coherence)

**C5: Readout error asymmetry fixed**
- Removed arbitrary `p1_given_0 = (1 - ro_fid) * 0.5` factor
- Now uses symmetric model: `p_error = (1 - ro_fid) / 2` for both directions
- Honest about available calibration data (single fidelity number)

**C6: Shot noise seed fixed**
- `seed_simulator=42` was hardcoded for all evaluations
- Same parameters always returned identical counts — not representative of QPU
- Fix: unique-but-reproducible seed per circuit via `hash((i, ci))`

**C7: Fermi-Hubbard + Heisenberg added to benchmarks (8 new configs)**
- Fermi-Hubbard 1×2 lattice (4q): noiseless, controlled, noisy, QPU
- Heisenberg XXZ 2×2 lattice (4q): noiseless, controlled, noisy, QPU
- Uses validated optimizer stack (L-BFGS-B + parameter_shift + adiabatic for FH)

**Q3: CircuitSubmissionWorkflow.get_required_plugins() fixed**
- Now declares hamiltonian and ansatz as required (were missing)

**Q4: VQAWorkflow dead class removed**
- Was a stub raising NotImplementedError, discoverable by registry
- Removed to prevent misleading `mode: vqa` config failures

### Benchmark config matrix (28 total)

| Model | Noiseless | Controlled | Noisy | QPU |
|-------|-----------|------------|-------|-----|
| TFIM 2q | ✓ | ✓ | ✓ | ✓ |
| TFIM 4q | ✓ | ✓ | ✓ | ✓ |
| TFIM 8q | ✓ | ✓ | ✓ | ✓ |
| H₂ 4q | ✓ (fixed) | ✓ | ✓ | ✓ |
| QAOA 8q | ✓ | ✓ | ✓ | ✓ |
| FH 4q (new) | ✓ | ✓ | ✓ | ✓ |
| Heis 4q (new) | ✓ | ✓ | ✓ | ✓ |


## 1.0.0b2 (2026-03-26)

### Phase 3 — Performance & QPU Integration

**Batched gradient computation**
- GradientStrategy ABC extended: `supports_batching`, `build_shifted_params()`, `assemble_gradient()`
- ParameterShift and FiniteDifference implement batched mode
- Workflow auto-detects batching support, submits all shifted circuits in one `sim.run()` call
- Aer distributes batch across available GPUs automatically
- For 64 params on 8 GPUs: ~8× speedup (128 sequential → 16 per GPU)
- Prints `[BATCHED]` in output when active; sequential fallback preserved

**AI/ML training data export**
- `src/lumi_hpc_qc/data/export.py` — flatten experiment JSON to CSV
- Per-iteration mode: every VQE step with energy, params, gradient norm
- Summary mode: one row per experiment with best energy, error, timing
- CLI: `python -m lumi_hpc_qc.data.export results/*.json --output training_data.csv`

**CircuitSubmissionWorkflow**
- Execute circuits without optimization loop
- Supports direct submission to any backend (GPU sim, QPU)
- Reports energy from counts (QPU) or statevector (sim)
- Useful for benchmarking, pre-optimized circuit execution, QPU testing

**IQM Q50 backend (FiQCI middleware)**
- Updated to use SLURM `q_fiqci` partition — no API token needed
- Reads `Q50_CORTEX_URL` from FiQCI environment (set by module)
- Correct sbatch pattern: `--partition=q_fiqci --mem-per-cpu=2G`
- `_expectation_from_counts()` for shot-based energy evaluation

**Q50 experiment configs**
- `q50_byo_tfim_2q.yaml`: 2-qubit TFIM on QB6-QB7 (CZ: 99.73%)
- `q50_byo_tfim_4q.yaml`: 4-qubit TFIM chain QB6-QB7-QB13-QB12
- Qubit selection based on March 26 2026 calibration data
- Avoids QB20, QB32, QB44, QB54 (low fidelity / missing)
- SU2 reps=1 (shallow circuits), SPSA/COBYLA (noise-tolerant)
- Calibration set ID: ce78e408-d231-4ce4-afea-898f06da818b

**Infrastructure**
- Removed `#SBATCH --account` from all scripts (uses SBATCH_ACCOUNT from env.sh)
- Fixed `SLURM_SUBMIT_DIR` sourcing for env.sh (BASH_SOURCE breaks in SLURM spool)
- Fixed `SLURM_MEM_PER_NODE` leak from Mode B controller to child jobs
- Restored `CHILD_SCRIPT_DIR` after overly greedy sed
- Reverted Mode B controller to `small` partition (standard bills full node)


## 1.0.0b1 (2026-03-26)

### Phase 2 — Production Resilience

**Step 2.1: Checkpoint/Resume**
- VQE workflow refactored into `_setup()` / `_initialize_params()` / `_optimize()`
- `resume()` rebuilds pipeline and continues from saved parameters
- Atomic checkpoint writes (write .tmp, rename) — no corruption on crash
- Auto-cleanup keeps latest checkpoint after successful completion
- Tested on LUMI: job 17012216, BYO TFIM 8q resumed from iter 40 → 0.052% error

**Step 2.2: New Backends**
- `aer_cpu` — MPS simulation for `standard` partition, scales to 100+ qubits
- `iqm_qpu` — VTT Q50 stub with `expectation_from_counts()` for shot-based evaluation

**Step 2.3: Mode B Controller**
- Pure bash controller (`tests/mode_b_controller.sh`) — no container, no GPU
- Runs on `small` partition (1 core, 1GB RAM)
- Submits/monitors GPU child jobs via sbatch/squeue
- Auto-retry from checkpoint on FAILED/TIMEOUT (configurable max retries)
- Launcher: `./tests/launch_mode_b.sh configs/experiment.yaml --walltime 02:00:00`

**Step 2.5: Error Mitigation**
- `zne` — Zero-noise extrapolation stub (plugin discovered, validates config)

**Infrastructure**
- `env.sh` — single file for all container/wrapper/account paths
- All sbatch scripts source `env.sh` — zero hardcoded absolute paths
- Separate CPU and GPU wrapper support (`HPCQC_CPU_WRAPPER`, `HPCQC_GPU_WRAPPER`)
- `MANIFEST` with sha256 hashes for every file
- `vqa_summary.sh` — descriptive model names, experiment IDs, error checking

## 1.0.0a1 (2026-03-26)

### Phase 1 — Foundation + All 5 Models

**Architecture** — 5-layer design, strict downward dependencies:
1. CLI (config, SLURM submission, status)
2. Orchestration (workflow, scheduler, checkpoint, controller)
3. Backends (Aer GPU — pluggable)
4. Plugins (6 sub-packages, auto-discovery)
5. Data (experiment tracker, provenance, timing, result store)

**Plugins implemented:**
- Hamiltonians: fermi_hubbard, heisenberg, qaoa_maxcut, molecular, byo
- Ansatze: hva, su2, qaoa, uccsd, byo
- Optimizers: l_bfgs_b, cobyla, spsa
- Gradients: parameter_shift, finite_difference (central, eps=0.1)
- Initializers: random, zero, adiabatic (GD + backtracking line search)

**LUMI test results (standard-g, MI250X, double precision):**

| Model | Best E | Exact E | Error | SLURM Job |
|---|---|---|---|---|
| BYO TFIM 8q (SU2) | -7.637 | -7.641 | **0.045%** ✓ | 17007219 |
| QAOA MaxCut 12q (COBYLA) | -13.01 | -14.00 | **7.06%** ✓ | 17008294 |
| FH 2×3 12q (HVA+adiabatic) | -1.97 | -5.78 | 65.9% | 17008291 |
| Heisenberg 3×4 12q (HVA+adiabatic) | -1.87 | -26.77 | 93.0% | 17008292 |
| H2 UCCSD 4q | -1.24 | -1.86 | 33.0% | 17008293 |

**Known research-level issues:**
- FH/Heisenberg: HVA 3-layer insufficient for 2D strongly correlated systems
- H2 UCCSD: qiskit-nature decomposition breaks parameter binding
- These are ansatz expressibility / qiskit version issues, not framework bugs

**Container:** qiskit 2.3.0, qiskit-aer 0.17.2, qiskit-nature 0.7.2, numpy 2.4.3, scipy 1.17.1
**Platform:** LUMI (CSC Finland), AMD MI250X GPUs, Singularity container
