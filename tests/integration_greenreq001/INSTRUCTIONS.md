# GREEN-REQ-001 — Container Integration Test Instructions

## For Running on LUMI After Container Rebuild Delivery

**Date:** March 29, 2026  
**Container Under Test:** `07-patchelf-fix.sif` — 69 packages (SLURM build job 17079529)  
**Test Window:** 48 hours from delivery  
**Delivered By:** Team Green (GREEN-RESP-001-DELIVERY-v1.0)

---

## PREREQUISITES

Before running any tests, you need:

1. **SSH access to LUMI** (UAN login node)
2. **The HPCQC project directory** deployed at `~/HPCQC` (or wherever your repo lives)
3. **The rebuilt container** — Team Green delivered it at:
   ```
   /flash/project_462001289/mucciard/CSC_QT_simulations_container_builder/ccpe-extensions-cray-qiskit-aer-patch/output/07-patchelf-fix.sif
   ```
4. **The test files** from `tests/integration_greenreq001/` copied into your project

---

## STEP 0: UPDATE env.sh TO POINT TO THE NEW CONTAINER

The rebuilt container may be at a different path than your current `env.sh` expects. Check and update if needed:

```bash
ssh lumi                          # or however you connect
cd ~/HPCQC                        # your project root

# Check current container path
grep HPCQC_GPU_CONTAINER env.sh
```

If the path in `env.sh` doesn't match the delivery location, update it. The delivery report says the container is at:

```
/flash/project_462001289/mucciard/CSC_QT_simulations_container_builder/ccpe-extensions-cray-qiskit-aer-patch/output/07-patchelf-fix.sif
```

Your existing `env.sh` may point to:

```
/flash/project_462001289/mucciard/ccpe-extensions-cray-qiskit-aer-patch/output/07-patchelf-fix.sif
```

**Option A — Update env.sh permanently:**
```bash
# Edit env.sh and change the HPCQC_GPU_CONTAINER path
vi env.sh
```

**Option B — Override for this session only:**
```bash
export HPCQC_GPU_CONTAINER="/flash/project_462001289/mucciard/CSC_QT_simulations_container_builder/ccpe-extensions-cray-qiskit-aer-patch/output/07-patchelf-fix.sif"
```

**Verify the container exists:**
```bash
ls -lh "${HPCQC_GPU_CONTAINER}"
# Should show ~11 GB file
```

Also add PYTHONHASHSEED=0 to env.sh if not already present (Phase A item 3):
```bash
# Add this line to env.sh if not already there:
echo 'export PYTHONHASHSEED=0' >> env.sh
```

---

## STEP 1: COPY THE TEST FILES TO LUMI

The integration test suite consists of 7 files. Copy them into your project:

```bash
# On LUMI, from your project root:
mkdir -p tests/integration_greenreq001

# Then copy these files into tests/integration_greenreq001/:
#   test_01_packages_and_versions.py
#   test_02_aer_gpu_and_vqe.py
#   test_03_checkpoint_and_determinism.py
#   slurm_test_01.sh
#   slurm_test_02.sh
#   slurm_test_03.sh
#   launch_all.sh

# Make scripts executable
chmod +x tests/integration_greenreq001/*.sh
```

---

## STEP 2: CREATE SLURM LOG DIRECTORY

```bash
mkdir -p slurm_logs
```

---

## STEP 3: RUN ALL TESTS

**The easy way — launch all 3 tests at once:**

```bash
cd ~/HPCQC                        # project root
source env.sh                     # load container paths + SLURM account
./tests/integration_greenreq001/launch_all.sh
```

This submits 3 SLURM jobs to the `standard-g` partition. You'll see output like:

```
GREEN-REQ-001 — Container Integration Test Suite
==========================================================
  Date:      2026-03-29T...
  Project:   /users/mucciard/HPCQC
  Container: /flash/.../07-patchelf-fix.sif
  Account:   project_462001289
==========================================================

Submitting integration tests...

  Test 1 (Packages & Versions):     job 17095001
  Test 2 (Aer GPU + VQE):           job 17095002
  Test 3 (Checkpoint + Determinism): job 17095003
```

**Or submit individually:**

```bash
cd ~/HPCQC
source env.sh

# Test 1 only (fast — ~1 min)
sbatch tests/integration_greenreq001/slurm_test_01.sh

# Test 2 only (medium — ~10 min)
sbatch tests/integration_greenreq001/slurm_test_02.sh

# Test 3 only (medium — ~10 min)
sbatch tests/integration_greenreq001/slurm_test_03.sh
```

---

## STEP 4: MONITOR

```bash
# Watch job queue
squeue --me

# Wait for all 3 to show "COMPLETED"
watch -n 10 squeue --me
```

Expected completion times:

| Test | SLURM Time Limit | Expected Actual |
|------|------------------|-----------------|
| Test 1 (packages) | 5 min | ~1 min |
| Test 2 (Aer GPU + VQE) | 20 min | ~5–10 min |
| Test 3 (checkpoint) | 20 min | ~5–10 min |

---

## STEP 5: CHECK RESULTS

**Quick pass/fail summary:**

```bash
grep -E 'PASS|FAIL|Exit code' slurm_logs/greenreq001_*.o*
```

You want to see `Exit code: 0` for all three and no `[FAIL]` lines.

**Detailed output for each test:**

```bash
# Test 1: Package imports + version checks
cat slurm_logs/greenreq001_t01_packages.o*

# Test 2: Aer GPU + VQE convergence
cat slurm_logs/greenreq001_t02_aer_vqe.o*

# Test 3: Checkpoint/resume + determinism
cat slurm_logs/greenreq001_t03_checkpoint.o*
```

**Check for errors in stderr:**

```bash
cat slurm_logs/greenreq001_*.e*
# Should be empty or contain only harmless warnings
```

---

## WHAT EACH TEST DOES

### Test 1: Packages & Versions (`test_01_packages_and_versions.py`)

**Purpose:** Verify all 8 new packages import correctly, numpy/pandas versions are as expected, numpy operations produce correct results, and the existing framework is unbroken.

**15 checks across 5 sections:**

| Section | Checks | What It Tests |
|---------|--------|---------------|
| Version checks | numpy 2.2.6, pandas 2.3.3, scipy 1.17.1 | Downgrades didn't break version expectations |
| numpy regression | polyfit, linalg.norm, RandomState determinism, array ops | Our code's numpy calls work on 2.2.6 |
| Group A imports | pyarrow + schema, mitiq + ZNE, scikit-learn + RF, jsonschema + validate | All 4 required packages functional |
| Group B imports | QCut + find_cuts, pymetis + part_graph, stim + Circuit, pymatching + Matching | All 3 recommended packages functional |
| Framework regression | All 5 layers import, plugin discovery (≥19 plugins) | Container rebuild didn't break existing code |

**Does NOT import qiskit_aer** (to avoid MPI_Init_thread conflict with mitiq in the same process).

**Expected output:** `ALL 15 CHECKS PASSED`

---

### Test 2: Aer GPU + VQE Convergence (`test_02_aer_gpu_and_vqe.py`)

**Purpose:** Verify the Aer GPU backend (custom ROCm fork) works correctly for both statevector and shot-based simulation, and that a full VQE workflow converges.

**4 sub-tests:**

| Sub-test | What It Tests | Pass Criterion |
|----------|---------------|----------------|
| 2a: Aer GPU statevector | 4-qubit GHZ state, verify amplitudes | \|0000⟩ and \|1111⟩ amplitudes ≈ 1/√2 |
| 2b: density_matrix shots | Bell state, 4096 shots, verify statistics | ~50/50 between \|00⟩ and \|11⟩ |
| 2c: VQE convergence | BYO TFIM 8q, L-BFGS-B, 200 iters | Relative error < 0.1% |
| 2d: Export pipeline | JSON result → CSV export | CSV written with >0 rows |

**Does NOT import mitiq** (separate process from test 1 to avoid MPI conflict).

**Expected output:** All 4 sub-tests pass, VQE error < 0.1%

---

### Test 3: Checkpoint/Resume + Determinism (`test_03_checkpoint_and_determinism.py`)

**Purpose:** Verify checkpoint/resume works on the new container and PYTHONHASHSEED=0 produces deterministic behavior.

**2 sub-tests:**

| Sub-test | What It Tests | Pass Criterion |
|----------|---------------|----------------|
| 3a: PYTHONHASHSEED | Verify PYTHONHASHSEED=0 is set, dict ordering deterministic | Environment variable set |
| 3b: Checkpoint/resume | Run VQE for 30 iters → checkpoint → resume for 100 more | Phase 2 energy ≤ Phase 1 energy |

**Expected output:** Both sub-tests pass, energy improves after resume

---

## INTERPRETING RESULTS

### All 3 tests pass (exit code 0, no FAIL lines):

**Container is promoted to production.** Report to Team Green:

> "All 3 integration tests passed. Container accepted. Promoting to production."

Update `env.sh` permanently to point to the new container (if you used Option B in Step 0).

### Test 1 fails:

**Likely cause:** A new package didn't install correctly or a numpy downgrade broke something.

- Check which specific check failed in the output
- If it's a numpy issue: report the exact operation that failed and the error message
- If it's a new package import failure: report which package and the traceback

### Test 2 fails:

**Likely cause:** Aer GPU backend is broken by the container rebuild.

- If test 2a (statevector) fails: the Aer GPU core is broken — **request rollback immediately**
- If test 2b (density_matrix) fails: shot-based path broken — may be related to numpy downgrade
- If test 2c (VQE) fails with >1% error: could be numpy affecting optimization convergence — report but don't rollback immediately (may need investigation)
- If test 2d (export) fails: application-level issue, not container — investigate separately

### Test 3 fails:

**Likely cause:** Checkpoint serialization broken by numpy version change (numpy arrays in checkpoints).

- Report the specific error — is it a load failure or an energy regression?
- If checkpoint can't be loaded: numpy 2.2.6 may serialize arrays differently than 2.4.3 — report to Team Green
- If energy regresses: likely an optimizer issue, not container

---

## ROLLBACK (IF NEEDED)

If any critical test fails and traces to container behavior:

```bash
# Tell Team Green to rollback
# They execute:
cp output/07-patchelf-fix.sif.bak output/07-patchelf-fix.sif
```

Then revert `env.sh` to point to the previous container path.

---

## AFTER ALL TESTS PASS

1. Update `env.sh` to permanently point to the new container
2. Report results to Team Green (close the 48-hour window)
3. Report results to Team Red (container is production, Phase A can proceed on new container)
4. Begin Phase A implementation:
   - Integrate Team Red unit tests
   - Fix launcher script (add FH/Heisenberg, noiseless_controlled)
   - PYTHONHASHSEED=0 already set (Step 0)
   - Verify ham_meta field naming consistency

---

## FILE INVENTORY

```
tests/integration_greenreq001/
├── launch_all.sh                          # Master launcher — submits all 3 SLURM jobs
├── slurm_test_01.sh                       # SLURM: package imports + versions
├── slurm_test_02.sh                       # SLURM: Aer GPU + VQE convergence
├── slurm_test_03.sh                       # SLURM: checkpoint/resume + determinism
├── test_01_packages_and_versions.py       # Python: 15 checks across 5 sections
├── test_02_aer_gpu_and_vqe.py             # Python: Aer GPU + VQE + export
└── test_03_checkpoint_and_determinism.py   # Python: checkpoint/resume + PYTHONHASHSEED
```

All SLURM output goes to `slurm_logs/greenreq001_*.o<JOBID>` and `slurm_logs/greenreq001_*.e<JOBID>`.
