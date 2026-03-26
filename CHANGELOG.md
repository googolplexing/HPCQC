<!-- Copyright (c) 2026 Michael Mucciardi -->
<!-- SPDX-License-Identifier: SSPL-1.0 -->

# Changelog

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
