<!-- Copyright (c) 2026 Michael Mucciardi -->
<!-- SPDX-License-Identifier: SSPL-1.0 -->

# Changelog

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

## 1.0.0rc1 (2026-03-31)

### Release Candidate 1 — Feature-Complete for Q50 Benchmark Campaign

**V19: Measurement stats capture end-to-end (RED-RESP-V19-v1.0)**
- `capture_measurement_stats` and `measurement_stats_interval` config fields in `types.py`
- Config loader reads from both top-level YAML and `data:` section
- `aer_gpu.py`: captures per-Pauli-group raw counts (labels, basis_rotations, counts,
  group_expectation, shots) after `expectation_from_grouped_counts()` returns
- `workflow.py`: passes capture flag via `CircuitJob.metadata`, writes JSONL sidecar
  via `ExperimentTracker.write_measurement_stats()`
- `experiment.py`: sidecar JSONL lifecycle — path setup in `start()`, interval-aware
  append-only writing (crash-safe), sidecar reference in final result JSON
- `export.py`: flat string array `measurement_stats` dataset in HDF5 (each element is
  one JSONL line — self-describing JSON, no subgroup hierarchy). Attributes:
  `grouping_algorithm`, `interval`, `num_entries`
- New `data/tools/` subpackage: `strip_basis_rotations` (lossless removal, ~30% size
  reduction) and `reconstruct_basis_rotations` (rebuild from Pauli groups + QWC algorithm)

**HDF5 human-readable group naming (RED-RESP-V19 Modifications 2+3)**
- All HDF5 experiment groups now use `{model}-{qubits:02d}q-{mode}-{id_short}-seed{seed:04d}`
- Dashes between fields, underscores within fields
- Sort order: model → qubit count → noise mode → experiment → seed
- Experiment UUID stored as group attribute for programmatic lookup
- Applies to all `export_hdf5()` output, not just V19 experiments
- Backward compatible: old UUID-based group names still readable

**Validation criteria closure**
- V1–V8 (Phase B): 37/37 PASSED — SLURM job 17094955
- VC1–VC9 (Phase C): 48/48 PASSED — SLURM job 17095335
- V15–V18 (Phase D): 51/51 PASSED — SLURM job 17117032
- V19 (measurement stats): 24/24 PASSED — SLURM job 17131462
- V20 (QPY integrity): PASSED — SLURM job 17118264
- Cross-implementation (§7.5): PASSED — SLURM job 17127770
- Phase D regression post-V19: 51/51 PASSED — SLURM job 17131652
- Phase D validation post-V19: 35/35 PASSED — SLURM job 17131653

**Deferred to pre-QPU (non-blocking for RC1)**
- Parameter binding verification after transpilation
- Thermal relaxation PadDelay for idle qubits
- Crosstalk calibration (OQ9) for multiplexed QPU execution

---

## 1.0.0b7 (2026-03-31)

### Phase D — Multi-Format Export, Schema v2, Quality Gates (RED-SPEC-001)

**Multi-format export pipeline (`data/export.py` — extended)**
- Parquet export via pyarrow: `list<double>` for parameters, full Arrow schema, Snappy compression
- HDF5 export via h5py: `/experiments/{id}/` hierarchy with energy_trajectory, param_trajectory,
  gradient_norms datasets and metadata attributes group
- JSONL export: one JSON object per iteration, append-friendly, Hugging Face datasets compatible
- NPZ export: per-experiment compressed numpy arrays with metadata JSON sidecar
- CSV export enriched with Phase D columns: noiseless_tier, circuit_depth_pre, cx_count_pre,
  coupling_map_source, mitigation_readout, schema_version
- `export_all()` convenience wrapper writes all 6 formats at once; graceful skip if
  pyarrow/h5py unavailable
- CLI: `python -m lumi_hpc_qc.data.export results/*.json --output dataset/ --format all`

**Schema v2.0.0 (`data/schema.py` — new)**
- Python-native `validate_record()` — no jsonschema dependency; returns list of error strings
- JSON Schema dict for external tooling
- `is_v1_record()` and `upgrade_v1_to_v2()` for backward-compatible record migration
- v1 records upgraded safely: Phase D fields added with safe defaults, scientific values unchanged

**Data quality gate (`data/quality.py` — new)**
- `QualityGate.run(record)` — 5 pre-write checks: completeness, consistency, convergence,
  energy_bound, iter_budget
- Never raises: gate failures produce `quality_report` with `passed=False`, result still written
- Called automatically in `ExperimentTracker.finalize()` before JSON write

**ExperimentRecord metadata enrichment (`types.py` — extended)**
- `HamiltonianMetadata`: `spectral_gap` (E1−E0 via scipy sparse eigensolver, None for >16q),
  `hamiltonian_locality` (max Pauli weight)
- `AnsatzMetadata`: `pre_transpilation_depth`, `pre_transpilation_cx_count`
- `ExperimentRecord`: `schema_version`, `noiseless_tier`, `error_mitigation_applied`,
  `per_placement_results`, `quality_report`
- `workflow.py` enriches ham_meta with locality+spectral_gap post-build; captures
  pre-transpilation metrics before transpile()

**QPY circuit export (`scripts/qpy_export.py` — new)**
- Rebuilds optimised ansatz circuit from result JSON at best parameters
- Serialises to QPY via `qiskit.qpy.dump()`; verifies round-trip (V20)
- SLURM script: `tests/slurm_qpy_export.sh`
- V20 PASS: job 17118264 (num_qubits=4, depth=11, num_parameters=0 — fully bound)

**Cross-implementation validation (`scripts/cross_impl_validation.py` — new)**
- Reference 1: pure numpy tensor products — no Qiskit, no HPCQC
- Reference 2: BYO Hamiltonian plugin via SparsePauliOp
- Reference 3: numpy vs plugin consistency (threshold 1e-10)
- SLURM script: `tests/slurm_cross_impl.sh`
- ALL PASS: job 17127770 (|diff| 3.14e-09 vs framework, 0.00e+00 between references)

**Calibration file fix**
- All configs updated from `q50_calibration_20260326.json` (13-qubit stub, 2293 bytes)
  to `q50_calibration_20260330.json` (53-qubit full calibration, 14295 bytes)
- 15 files fixed: `configs/generated/` and `configs/`

**Qiskit 2.3.0 transpiler bug fix (`backends/aer_gpu.py`)**
- `_FakeTarget._coupling_map` → `_coupling_graph` rename in Qiskit 2.3.0 broke
  shot-based transpilation when noise model is active
- Fix: pass `basis_gates` explicitly to `transpile()` to bypass `_FakeTarget` lookup

**SLURM validation**
- `tests/slurm_phase_d.sh` — 35/35 PASSED (jobs 17116765, 17116672)
- `tests/slurm_phase_d_export.sh` — 51/51 PASSED (job 17117032)
- `tests/check_phase_d_packages.py` — jsonschema 4.26.0, pyarrow 23.0.1, h5py 3.16.0 confirmed
- **86/86 total SLURM-verified checks**

---

## 1.0.0b6 (2026-03-30)

### Phase C — Error Mitigation, Multiplexed QPU, Statistical Characterisation (RED-SPEC-001)

**Readout error mitigation (`plugins/error_mitigation/readout.py` — new)**
- Tensor product calibration matrix construction from per-qubit readout fidelities
- O(N × 2^N) matrix inversion via numpy; symmetric error model
- Activated via `error_mitigation: readout_correction: true` in config

**Zero Noise Extrapolation (`plugins/error_mitigation/zne.py` — new, 222 lines)**
- Replaces Phase B stub with full mitiq integration
- Gate folding at configurable scale factors (default: [1, 3, 5])
- Linear, polynomial, and exponential extrapolation options
- Lazy mitiq import (MPI_Init_thread safety)
- `apply_every: N` parameter — apply ZNE every Nth evaluation, raw energy for intermediate steps
- ZNE always applied during gradient computation steps regardless of `apply_every`

**Multiplexed QPU circuit builder (`plugins/placement/multiplexer.py` — new)**
- 53-qubit composite circuit construction for simultaneous multi-placement execution
- Per-placement bitstring demultiplexing from combined measurement results
- Per-placement energy computation and metadata assembly
- Placement metadata includes physical qubit IDs, topology shape, calibration parameters

**Multi-seed sweep infrastructure**
- `scripts/generate_seed_sweep.py` — generates N config variants from base config,
  differing only in `initializer_params.seed`
- SLURM launcher scripts for TFIM 2q and 4q 20-seed sweeps
- Reproducibility comparison script (`scripts/reproducibility_check.py`)

**Seed sweep results (LUMI jobs 17095732–17095751, 17095916–17095935)**
- TFIM 2q: 20/20 seeds exact convergence (0.000% error, mean 15.2 iterations)
- TFIM 4q: 3-tier convergence structure:
  - Tier 1 (<0.01%): 5/20 seeds — global minimum found
  - Tier 2 (0.01–1%): 11/20 seeds — sub-optimal saddle points
  - Tier 3 (>1%): 4/20 seeds — trapped at E≈−4.659 (2.098% error)
- Tier 3 adiabatic re-runs (RED-EXPERIMENT-TIER3-ADIABATIC-v1.0): all 4 seeds
  escaped Tier 3 but converged to consistent Tier 2 saddle at E=−4.734 (0.516%)
  — adiabatic initialiser is deterministic in output; eliminates Tier 3 trapping

**48/48 SLURM-verified checks (job 17095335)**

---

## 1.0.0b5 (2026-03-30)

### Phase B — Parameterised Noise, Topology-Noiseless, Circuit Metrics (RED-SPEC-001)

**Parameterised noise model (`backends/noise_model.py` — extended)**
- `noise_channels` YAML config for selective channel activation:
  `single_qubit_depolarizing`, `two_qubit_depolarizing`, `t1_relaxation`,
  `t2_dephasing`, `readout_error`
- Per-qubit parameters sourced from Q50 calibration JSON (T1, T2, gate errors,
  readout fidelity)
- Backward compatible: existing configs without `noise_channels` use all channels

**Topology-noiseless mode**
- `coupling_map_source` independent of noise model
- Supports: `calibration` (Q50 connectivity), `full` (all-to-all), `file` (custom YAML),
  `linear`, `grid`, `heavy_hex`
- Enables topology-isolated testing: routing overhead measured without noise overhead

**Circuit metrics in ExperimentRecord (`types.py` — extended)**
- `CircuitMetrics` dataclass: pre/post transpilation depth, gate count, CX count,
  SWAP count, coupling map source, coupling map edges, transpiler optimisation level

**Config generator (`scripts/generate_configs.py`)**
- 7 base model configs × 13 mode templates → 91 benchmark configs
- 13 committed representative configs: all modes for TFIM 4q

**Placement solver (`plugins/placement/solver.py` — new, 288 lines)**
- rustworkx graph operations + cvxpy optimisation
- Connectivity-aware qubit subgraph selection (greedy connected subgraph)
- 9 non-overlapping 4-qubit placements on Q50 (68% utilisation)

**Q50 calibration update**
- Full 53-qubit calibration (`q50_calibration_20260330.json`, 14295 bytes)
  replacing 13-qubit stub (`q50_calibration_20260326.json`, 2293 bytes)

**37/37 SLURM-verified checks (job 17095243)**

---

## 1.0.0b4 (2026-03-30)

### Phase A — Unit Tests, Launcher, Ansatz Validation (RED-SPEC-001)

**Team Red unit tests integrated (RED-SPEC-001-v1.1 §C)**
- 5 Hamiltonian verification tests: TFIM ZZ+X terms, Fermi-Hubbard hopping+interaction,
  Heisenberg XXX exchange, QAOA MaxCut cost Hamiltonian, H₂ Jordan-Wigner 4-qubit encoding
- 3 F1 Pauli measurement tests: shot-based ⟨H⟩ within 3σ of exact for X, Y, Z terms;
  multi-group observable; identity term handling
- 3 decomposition equivalence tests: SU2, UCCSD, QAOA unitaries preserved through
  `decompose_for_aer()` with rtol=1e-10

**Launcher updates**
- 7 models: added `fh_4q` and `heis_4q` to MODELS list
- 4 modes: added `noiseless_controlled` and `topology_noiseless` as supported modes
- Updated help text

**ham_meta field naming standardised**
- `pauli_term_count` → `num_pauli_terms`
- `model_params` → `physical_params`

**Ansatz validation hardening**
- QAOA: edge index bounds check against num_qubits
- UCCSD: electrons-vs-qubits consistency check
- BYO: QASM qubit count validation
- 13 lines of defensive validation across 3 plugins

**Bug fixes**
- F-3: HVA 2-qubit entanglement layer fixed (was silently dropping last qubit)
- F-1: TFIM 2q exact energy corrected (reference was −√5, not −2.0)

**Environment**
- `PYTHONHASHSEED=0` added to `env.sh` — deterministic seed computation for C6 fix

**31/31 tests passed (SLURM job 17093900)**

## 1.0.0b3 (2026-03-28)

### Team Red Code Review — Critical Fixes

**F1 (CRITICAL): Pauli basis rotation for shot-based measurement**
- New `backends/pauli_measurement.py` (263 lines) — correct X/Y/Z handling via basis rotation
  (H gate for X positions, Sdg+H for Y positions) before Z-basis measurement
- `aer_gpu.py` updated: shot-based path uses `build_measurement_circuits` + `expectation_from_grouped_counts`
- `iqm_qpu.py` updated: old broken `_expectation_from_counts` (Z-only parity) fully removed,
  replaced with `pauli_measurement` module imports
- Impact: every Hamiltonian with X or Y terms (TFIM, Heisenberg, molecular) now produces
  correct shot-based and QPU results

**F2 (CRITICAL): VQE eval_energy routed through Backend.run_circuits()**
- `eval_energy()` and `eval_energy_batch()` now create `CircuitJob` and call `backend.run_circuits()`
- All direct `backend._sim.run()` bypass calls eliminated from VQE path
- Shot config (`shots: 4096`) now respected — noisy simulation actually applies noise
- Readout errors now applied in noisy simulation
- QPU execution path functional (no longer crashes on missing `_sim` attribute)

**C1: Topology-aware noisy simulation**
- Coupling map loaded from Q50 calibration data
- Transpilation respects physical qubit connectivity

**C4: Single-qubit noise double-counting removed**
- Thermal relaxation removed from single-qubit gate errors (depolarizing only from RB data)

**C5: Symmetric readout error model**
- Removed arbitrary `p1_given_0 = (1 - ro_fid) * 0.5` factor
- Now uses symmetric model: `p_error = (1 - ro_fid) / 2` for both directions
- Honest about available calibration data (single fidelity number)

**C6: Unique seed per circuit for realistic shot noise**
- `seed_simulator=42` was hardcoded for all evaluations (same params → identical counts)
- Fix: unique-but-reproducible seed per circuit via `hash((i, ci))`

**Q3: CircuitSubmissionWorkflow.get_required_plugins() fixed**
- Now declares hamiltonian and ansatz as required (were missing)

**Q4: VQAWorkflow dead class removed**
- Was a stub raising NotImplementedError, discoverable by registry

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
- MI250X distributes across GPUs automatically

**Adiabatic initialization**
- GD + backtracking line search over parameter-shift gradients
- Ramps coupling strength from 0 → target in configurable steps
- Warm-starts from a classically tractable regime
- L-BFGS-B + parameter_shift + adiabatic → 0.045% error on TFIM 8q (best result)

**AI/ML training data export**
- `data/export.py`: JSON → flat CSV (per-iteration and summary modes)
- Columns: experiment_id, model, ansatz, optimizer, energy, params, timing, error
- Designed for energy prediction, optimizer selection, circuit architecture search

**SLURM infrastructure**
- `SLURM_MEM_PER_NODE` leak fixed in Mode B controller
- `BASH_SOURCE` replaced with `SLURM_SUBMIT_DIR` (SLURM spool compatibility)
- Parentheses in bash comments removed (SLURM parser issue)
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
