# Testing Guide — lumi-hpc-qc

This document explains how to run every test suite in the project,
what each one checks, and how to interpret the output. Written for a
developer who is new to the project.

---

## Overview of test suites

| Suite | File | What it tests | Run time | Needs GPU? |
|---|---|---|---|---|
| Unit tests | `tests/unit/` | Python logic, no Qiskit | ~5s | No |
| Phase B validation | `tests/phase_b_validation.py` | Noise model, coupling maps, circuit metrics | ~30s | No |
| Phase C validation | `tests/phase_c_validation.py` | Readout mitigation, ZNE, multiplexed QPU | ~30s | No |
| **Phase D validation** | `tests/phase_d_validation.py` | Schema, quality gate, new metadata fields | ~5s | No |
| LUMI integration | `tests/slurm_phase_d.sh` | Same as above + container package check | ~2min | Yes (1 GPU) |

All suites print ✓ or ✗ per check and a final pass/fail summary.

---

## Running locally (no LUMI, no container)

The unit tests and all phase validation suites run on any Python 3.12+
with `qiskit` and `numpy` installed. They do not require a GPU, a
container, or a SLURM allocation.

```bash
cd ~/HPCQC   # repo root

# Unit tests (pytest)
python3 -m pytest tests/unit/ -v

# Phase B validation
python3 tests/phase_b_validation.py

# Phase C validation
python3 tests/phase_c_validation.py

# Phase D validation (new — Steps 1-3)
python3 tests/phase_d_validation.py
```

### Expected output — Phase D (local)

```
=================================================================
  Phase D Validation — Steps 1–3 (types, schema, quality gate)
  RED-SPEC-001 §8, V15-V20 (partial — export formats pending)
=================================================================

Group 1: types.py Phase D additions
  ✓  1/35  HamiltonianMetadata: spectral_gap field exists (default None)
  ✓  2/35  HamiltonianMetadata: hamiltonian_locality field exists (default 0)
  ...
  ✓ 14/35  compute_hamiltonian_locality: returns 0 on error (safe fallback)

Group 2: data/schema.py
  ✓ 15/35  validate_record: valid v2 record returns no errors
  ...
  ✓ 27/35  upgrade_v1_to_v2: upgraded record passes validate_record

Group 3: data/quality.py — QualityGate
  ✓ 28/35  QualityGate: valid record passes all five checks
  ...
  ✓ 32/35  QualityGate: does not raise on minimal/empty record

Group 4: ExperimentTracker integration
  ✓ 33/35  ExperimentTracker: _noiseless_tier attribute present
  ✓ 34/35  ExperimentTracker: _quality_report attribute present
  ✓ 35/35  ExperimentTracker: _error_mitigation_applied attribute present

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase D validation: 35/35 PASSED  ✓
```

**What it means:** All Phase D dataclass additions and the two new
modules (`schema.py`, `quality.py`) are working correctly. Old code
that doesn't know about the new fields is unaffected (backward
compatibility verified in checks 7–12).

---

## Running on LUMI via SLURM

LUMI runs are needed for anything that requires the GPU container
(actual Qiskit simulation, pyarrow Parquet writing, HDF5 writing,
QPY serialisation).

```bash
cd ~/HPCQC && source env.sh

# Submit Phase D validation
sbatch tests/slurm_phase_d.sh

# Watch the queue
squeue --me

# Check output when done (replace JOBID)
cat slurm_logs/phase_d_val.oJOBID
cat slurm_logs/phase_d_val.eJOBID   # should be empty or just an Lmod warning
```

### Expected SLURM output

```
=== Phase D Validation — LUMI container run ===
Job ID:    17096XXX
Node:      nid00XXXX
Container: /path/to/07-patchelf-fix.sif
Started:   Mon Mar 30 10:00:00 EEST 2026

[... same 35-check output as local ...]

Phase D validation: 35/35 PASSED  ✓

Finished: Mon Mar 30 10:01:30 EEST 2026
Exit code: 0

--- Container package check ---
jsonschema: 4.26.0
pyarrow:    23.0.1
h5py:       3.16.0
All Phase D export dependencies present.
```

**What the container check means:** Confirms that `jsonschema`,
`pyarrow`, and `h5py` are all available after the GREEN-REQ-001
container rebuild. Step 4 (export.py) requires all three of these.

---

## Running the full regression suite

Before tagging any new version, run all suites in sequence:

```bash
cd ~/HPCQC && source env.sh

# 1. Local unit tests (fastest — no SLURM needed)
python3 -m pytest tests/unit/ -v --tb=short

# 2. Local phase validations
python3 tests/phase_b_validation.py
python3 tests/phase_c_validation.py
python3 tests/phase_d_validation.py

# 3. Full LUMI integration (submit as SLURM jobs, wait for completion)
sbatch tests/slurm_phase_d.sh
```

All must show 0 failures before a version tag is pushed.

---

## What each group of Phase D checks verifies

### Group 1: types.py (checks 1–14)

These confirm that new optional fields added to `HamiltonianMetadata`,
`AnsatzMetadata`, and `ExperimentRecord` don't break existing code.
Every new field has a safe default (`None` or `0`) — code written
before Phase D that constructs these dataclasses without the new
fields will still work unchanged.

The `asdict()` checks (11–12) are especially important: the
`ExperimentTracker` uses `dataclasses.asdict()` to serialise results
to JSON. If a new field caused `asdict()` to fail, all experiment
results would be lost silently.

### Group 2: schema.py (checks 15–27)

These test the new `validate_record()` function, which is what
prevents bad data from being written to disk. Key things verified:

- A correct record produces zero errors (check 15)
- Each specific validation rule catches the right violation
  (checks 16–21)
- The v1→v2 upgrade path works: old results from before Phase D
  can be loaded and passed through the new pipeline (checks 22–27)

### Group 3: quality.py (checks 28–32)

These test the five quality gate checks individually. Each check
has both a "should pass" case and a "should fail" case tested.
Check 32 is specifically about resilience: the quality gate must
never crash the framework, even when given a completely empty record.

### Group 4: Integration (checks 33–35)

Confirms that `ExperimentTracker` (the class that actually runs
experiments) has the new Phase D attributes that wire the quality
gate and other new features into the VQE loop.

---

## Troubleshooting

**`ModuleNotFoundError: No module named 'lumi_hpc_qc'`**
Run from the repo root, not from inside `tests/`:
```bash
cd ~/HPCQC && python3 tests/phase_d_validation.py
```

**`ModuleNotFoundError: No module named 'qiskit'`**
Local environment doesn't have Qiskit. This is fine for Phase D
validation — it only needs stdlib + numpy. For Phase B/C validation
you need Qiskit; use the container via SLURM for those.

**Quality gate check fails unexpectedly**
Run with verbose output:
```python
from lumi_hpc_qc.data.quality import QualityGate
gate = QualityGate()
report = gate.run(record)
print(report["warnings"])   # shows exactly what failed and why
```

**SLURM job fails immediately**
Check the error log:
```bash
cat slurm_logs/phase_d_val.eJOBID
```
The most common cause is a missing `source env.sh` before `sbatch`.
