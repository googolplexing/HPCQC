<!-- Copyright (c) 2026 Michael Mucciardi -->
<!-- SPDX-License-Identifier: SSPL-1.0 -->

# Changelog

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
