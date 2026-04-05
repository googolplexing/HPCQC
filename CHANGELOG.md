<!-- Copyright (c) 2026 Michael Mucciardi -->
<!-- SPDX-License-Identifier: SSPL-1.0 -->

# Changelog

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

**New Test Sections (`tests/e7_sweep_validation.py`)**
- VE18: +2 checks — `exact_ground_energy` HDF5 attribute present and finite
- E7.11: +19 checks — LHS config parsing, grid expansion (10 samples, unique params,
  range validation), end-to-end sweep, HDF5 persistence (model_params, exact_ground_energy,
  key presence, range check), physics diversity (multiple distinct ground energies)

## 1.1.1 (2026-04-05)

### DSatur Packing, IQM Batch Submission, Multiprocessing Deadlock Fix (RED-RESP-PACKING-v1.0)

11 items, 97/97 validation checks on LUMI. Parquet schema: 63 columns (was 61).

**Item 1: Cache Key Fix (`sweep/twin_simulator.py`)**
- Noiseless dedup cache key now includes `obs_hash:num_qubits:topology_hash:env_name`
- Prevents silent collision when different Hamiltonians share topology
- Critical for DiagnosticTFIM coexistence with standard TFIM in same sweep

**Items 2–5: Sweep Engine Enhancements (`sweep/sweep_engine.py`, `sweep/execution_planner.py`)**
- E2/E7 Level 1 integration: TieredExecutionPlanner wired into SweepEngine
- `seed_offset` parameter in SweepExperimentConfig — enables non-overlapping seed
  ranges across SLURM jobs (ANIMLL multi-node submission)
- YAML `measurement_stats_interval` override via `dataclasses.replace()` on NoiseConfig
- VE19 amendment: 5 new assertions for interval override + preserve behavior

**Item 6: DSatur Optimal Packing (`sweep/placement_solver.py`)**
- `_pack_dsatur()` via `rx.graph_greedy_color` — provably optimal graph coloring
- Q50 4q star: 16 rounds (optimal) vs greedy 21 rounds (24% reduction)
- Default strategy changed from `"greedy"` to `"optimal"`
- Greedy remains available via `strategy: greedy` in YAML

**Item 7: IQM Batch Submission (`backends/iqm_qpu.py`)**
- `_submit_batch()` accepts `list[QuantumCircuit]` as single batch job
- Auto-chunking at VTT 200-circuit limit with correct boundary reassembly
- Test A (single batch ordering, 6/6) and Test B (multi-batch boundary, 7/7) — merge-blocking

**Item 8: Dynamic QB Mapping (`plugins/calibration_adapters/iqm_v2.py`)**
- QB32 deactivation documented — index gap at 31 (QB1→0 ... QB31→30, QB33→31 ... QB54→52)
- Defensive assertion: `len(qubit_names) == len(name_to_idx)`
- Mapping is calibration-set dependent — documented for future IQM topology changes

**Items 9–11: CZ Fidelity Columns (`data/sweep_export.py`)**
- `per_edge_cz_fidelity` confirmed populated end-to-end (HDF5 → Parquet)
- NEW: `placement_min_cz_fidelity` — `min(per_edge_cz_fidelity)`, noise-dominant edge
- NEW: `placement_avg_cz_fidelity` — `mean(per_edge_cz_fidelity)`

**Multiprocessing Deadlock Fix — Lesson #22 (`sweep/sweep_engine.py`)**
- CRITICAL: `mp.Pool` with fork copies locked C++ mutex state from parent's numpy BLAS,
  h5py, and Aer imports. Deadlock is scale-dependent — works at 8 workers, hangs at 100+.
- FIX: `_run_pool_subprocess()` serializes work items to pickle, launches fresh subprocess
  with no inherited C++ state, runs Pool there, reads results back.
- Proven on LUMI: Pool(100) → 100/100 success in 7.43s
- Trade-off: cross-battery noiseless dedup temporarily disabled (~1s extra per sweep).
  Restoration planned for v1.2.0 via two-subprocess pattern.
- All `mp.Pool` callers audited: `sweep_engine.py`, `execution_planner.py`, `eval_runner.py`

**Device Name Correction**
- `"VTT Q50 (Aalto Helmi)"` → `"Q50"` in calibration JSON
- Aalto Helmi is the 5-qubit machine, not Q50
- Matches VTT QX API, FiQCI SLURM, and CSC documentation

**New Test Files**
- `tests/test_v111_packing_batch.py` — Test A/B/C/D (19 checks)
- `tests/test_v111_validate.sh` — SLURM validation script (full node, OMP_NUM_THREADS=1)
- `tests/fork_test*.py` — multiprocessing deadlock diagnostic suite

## 1.1.0 (2026-04-03)

### Phase E — Sweep Engine + Noise Atlas Pipeline (RED-SPEC-002)

Complete YAML-to-Parquet pipeline for systematic noise characterization across
QPU topologies. 764 checks across 11 E-steps, zero failures. All 22 VE criteria satisfied.

**E1: Placement Solver (`sweep/placement_solver.py`, `topology_library.py`)**
- VF2 subgraph isomorphism via rustworkx — finds all valid physical qubit placements
  for arbitrary circuits on arbitrary QPU topologies
- 7 reference topologies (2q–8q): pair, chain, star, square
- Multi-device: calibration adapter registers devices, solver searches across all
- Scoring strategies: max_fidelity, max_connectivity, min_error, diverse
- Topology equivalence hashing (degree sequence canonical form)
- Q50 results: 487 4q placements (379 chain + 108 star), 3 distinct topology classes

**E2: Tiered Execution Planner (`sweep/execution_planner.py`)**
- CPU/GPU routing: ≤8q density_matrix → CPU, ≥10q → GPU
- 128-way CPU parallelism via multiprocessing.Pool (fork COW, no parent Aer)
- MPICH_GPU_SUPPORT_ENABLED=0 for CPU partition (permanent fix)

**E3: HDF5-First Writer (`data/hdf5_writer.py`)**
- Write-during-execution with WAL (write-ahead log) crash safety
- Atomic group writes: each result flushed immediately
- SWMR mode support (Lustre-dependent)
- Noiseless deduplication via HDF5 soft links

**E4: Twin Simulator (`sweep/twin_simulator.py`, `noise_configs.py`)**
- 11 noise environments per placement per calibration
- Tiered measurement stats: Tier A=5, Tier B=20, noise_full=10, noiseless=0
- Noiseless deduplication across calibrations (topology-dependent only)
- Per-placement noise model from calibration data
- SyntheticAdapter validates perturbation keys (ValueError on unknown)

**E5: BYO Circuits (`sweep/circuit_loader.py`, `eval_runner.py`)**
- Load circuits from QPY, QASM, or Python scripts
- Evaluation-only mode: no optimizer, single execution per config
- Connectivity extraction for placement solver integration

**E6a: Multi-Round Packing (`sweep/circuit_composer.py`, `demultiplexer.py`, `round_executor.py`)**
- Same-circuit packing: 379 chain placements → 64 rounds (9–10 per round)
- Non-overlapping qubits AND coupling edges verified
- Shot-based and exact (density matrix) execution modes

**E6b: Mixed-Experiment Packing (`sweep/mixed_packing.py`)**
- Different circuits from different experiments share QPU submissions
- MixedPacker: greedy non-overlapping round finder across experiment queues
- compose_mixed_round: heterogeneous circuits into device-width composite
- demux_mixed_counts: route results back to correct experiments
- Noisy validation: mixed ≈ independent under real Q50 calibration noise

**E7: Sweep Engine (`sweep/sweep_engine.py`)**
- YAML config → grid expansion → placement → twin battery → HDF5
- Cache-locality grouping: placements computed once, reused across seeds
- Cross-calibration noiseless deduplication
- Noise fingerprinting computed during execution (F1, F2, F5, F6, F8)
- Per-edge CZ fidelity extracted from placements

**E8: Sweep Export (`data/sweep_export.py`)**
- HDF5 → 61-column Parquet (RED-DIRECTIVE-E4-SCHEMA-v1.0 §4)
- Metadata-scan export: reads attributes only, O(N) in results
- Summary CSV for quick inspection
- Snappy compression

**E9: Synthetic Calibration CLI (`data/tools/perturb_calibration.py`)**
- 7 perturbation types: scale_t1/t2/readout/gate_fidelity, poison_qubit,
  uniform_noise, improve_all
- Batch generation for noise regime sweeps
- `_synthetic_metadata` provenance in every output JSON
- Physical constraints enforced (T2 ≤ 2*T1, readout clamped)

**E10: Validation + FiQCI Examples**
- FiQCI circuit builders: GHZ (3/4/5q), Bell (2q), Star (4q) in `examples/fiqci/`
- QPY round-trip via circuit_loader validated
- Physics: GHZ-3q ⟨ZZZ⟩=0, Bell ⟨ZZ⟩=1, Star-4q ⟨ZZZZ⟩=1
- Multi-calibration sweep: real + synthetic produce different energies

**New Plugin: TFIM Hamiltonian (`plugins/hamiltonians/tfim.py`)**
- Auto-discovered by registry: `name = "tfim"`
- H = -J Σ Z_i Z_j - g Σ X_i with configurable J, g, boundary conditions
- 1D chains and 2D grids

**Cross-Check Fixes (RED-RESP-V7-CROSS-CHECK-v1.0)**
- `measurement_stats_schedule: list[int] | None` in ExperimentConfig + capture logic
- Noise fingerprinting F1/F2/F5/F6/F8 computed during execution, stored as HDF5 attrs
- `per_edge_cz_fidelity` extracted from placement edges + calibration

**Validation (764 checks, zero failures)**
- E1: 70/70 (SLURM 17171430)
- E2: 36/36 (SLURM 17171581)
- E2.1: 15/15 (SLURM 17169665)
- E3: 39/39 (SLURM 17168540)
- E5: 43/43 (SLURM 17172442)
- E4: 71/71 (SLURM 17174016)
- E6a: 32/32 (SLURM 17174597)
- E7: 92/92 (SLURM 17175062)
- E8: 86/86 (SLURM 17179592)
- E9: 60/60 (SLURM 17179873)
- E10: 57/57 (SLURM 17180201)
- E6b: 50/50 (SLURM 17180507)
- V19 regression: 24/24, Phase D regression: 86/86

---

## 1.0.0b3 (2026-03-28)

### Team Red Code Review — Critical Fixes

**F1: Pauli basis rotation for shot-based measurement (CRITICAL)**
- `_expectation_from_counts()` was wrong for all non-Z Pauli terms (X, Y)
- X and Y operator contributions were silently dropped — only Z-parity computed
- Impact: all shot-based results (noisy simulation + QPU) scientifically invalid
- Fix: new `backends/pauli_measurement.py` module implements correct basis rotation
  - Groups qubit-wise commuting Pauli terms
  - Builds rotated measurement circuits (H for X, Sdg+H for Y positions)
  - All measurement in Z basis after rotation — parity computation now correct
- Both `aer_gpu.py` and `iqm_qpu.py` updated to use the new module

**F2: VQE workflow routed through Backend.run_circuits() (CRITICAL)**
- `eval_energy()` directly accessed `backend._sim.run(shots=0)`, bypassing:
  - The Backend abstract interface (violated dependency inversion)
  - Shot-based evaluation (config `shots: 4096` was ignored)
  - Readout noise (save_expectation_value skips readout errors)
  - QPU execution (crashed — IQM backend has no `_sim.run()`)
- Fix: eval_energy and eval_energy_batch create CircuitJob → backend.run_circuits()
  - Backend handles statevector vs shot-based routing internally
  - Noisy configs now actually apply readout noise
  - QPU path no longer crashes
  - Batched gradient works through same interface (no _sim check)

**C1: Topology-aware noisy simulation**
- Noisy Aer simulation now transpiles circuits to Q50 coupling map
- `noise_model.py` returns `(NoiseModel, CouplingMap)` tuple
- `aer_gpu.py` transpiles shot-based circuits to coupling map when noise model active
- Matches routing overhead of real QPU execution

**C2: Controlled benchmark comparison (7 new configs)**
- Added `*_noiseless_controlled.yaml` for all 7 models
- Uses same optimizer/gradient/reps as noisy configs (SPSA, no gradient, reduced reps)
- Enables 4-point comparison: ideal → controlled → noisy → QPU
- Each successive pair isolates exactly one variable

**C3: H₂ noiseless baseline fixed**
- Changed from COBYLA+random (33% error) to L-BFGS-B+parameter_shift+zero init
- UCCSD ansatz requires structured initialization, not random

**C4: Single-qubit noise double-counting removed**
- Noise model previously applied both depolarizing AND thermal relaxation
- RB-measured single_gate_error already includes coherence contributions
- Fix: depolarizing only (from RB data). Thermal relaxation removed.

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
