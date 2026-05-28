#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=w1_canary
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/w1_canary.o%j
#SBATCH --error=slurm_logs/w1_canary.e%j
#
# W1 Acceptance — 2-seed gate-2 canary on the W1 forkserver engine.
#
# Runs the W1.3-rewritten _execute_byo_group on the 2-seed canary YAML and
# asserts byte-match against the in-tree oracle banked at
# evidence/W1/gate2_canary/sha256_oracle.txt (RED-RESP-W1-PARALLELISM-AND-
# OOM-ROOTCAUSE-v1.4 §6 acceptance Criterion 2 / F3 "canary byte-match").
#
# Oracle SHAs (pin one arm of the 2-seed corpus):
#   seed 00: f5578984383107bc0e3f6eb57be7c8b5c980622f544af5cf1f3d10cdcdc82409
#   seed 01: 622721c61cb81e4a8fd313b34072173d638d18222c967dd7df1caf0f23ad1c9b
#
# The 40-seed aggregate byte-match (the full gate-2 reproduction) is W1.6 /
# Criterion 3; it uses a different SLURM script and a different walltime.
#
# Usage: sbatch tests/slurm_w1_canary.sh
# Expected: "W1 CANARY ACCEPTANCE: ALL CHECKS PASSED" on the last line and
# exit code 0.

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

CANARY_YAML="${HPCQC_ROOT}/examples/byo/floquet_dtc_q10_canary_2seed.yaml"
ORACLE="${HPCQC_ROOT}/evidence/W1/gate2_canary/sha256_oracle.txt"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/sweep_output_w1_canary_${SLURM_JOB_ID}"

echo "=== W1 Canary — 2-seed byte-match against in-tree oracle ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_CPU_CONTAINER}"
echo "YAML:      ${CANARY_YAML}"
echo "Oracle:    ${ORACLE}"
echo "Output:    ${OUTPUT_DIR}"
echo "Started:   $(date)"
echo ""

# ── Run the W1 engine on the 2-seed canary YAML ───────────────────────────
export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"

srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" python3 -c "
import sys, os
os.chdir('${HPCQC_ROOT}')
from lumi_hpc_qc.sweep.sweep_engine import run_sweep_from_yaml
result = run_sweep_from_yaml('${CANARY_YAML}')
print(f'Sweep finished: sweep_id={result.sweep_id} errors={len(result.errors)}')
sys.exit(0 if not result.errors else 1)
"

echo ""
echo "=== Byte-match verification ==="

# ── Locate engine outputs + assert byte-match against the oracle. The oracle
#    pins ONE arm of the canary (selected at banking time). The verifier
#    walks BOTH arms in the engine output and reports SHA matches; PASS iff
#    the same arm matches the oracle SHAs for both seeds. ──
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" python3 <<EOF
import hashlib, os, sys, glob

ORACLE = "${ORACLE}"
SUBMIT = "${SLURM_SUBMIT_DIR}"

# Parse oracle: SHA per (seed-index inferred from filename)
oracle_sha = {}
with open(ORACLE) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, path = line.split(None, 1)
        # canary_seed_00_instance_00_autocorr.dat -> seed_idx = 0
        base = os.path.basename(path)
        # seed_NN pattern
        import re
        m = re.search(r"seed_(\d+)_instance", base)
        if m:
            oracle_sha[int(m.group(1))] = sha
print(f"Oracle SHAs: {oracle_sha}")

# Find engine per-instance outputs. The engine writes
# {byo_dat_dir}/{script_stem}/{phys}/{env}/instance_NN_autocorr.dat
candidates = glob.glob(
    os.path.join(SUBMIT, "sweep_output*", "**", "instance_*_autocorr.dat"),
    recursive=True,
)
# Group by (arm, seed_index)
by_arm = {}
for path in candidates:
    parts = path.split(os.sep)
    # Find the env in the path (one of "noiseless" / "device_calibrated")
    env = None
    for p in parts:
        if p in ("noiseless", "device_calibrated"):
            env = p
            break
    if env is None:
        continue
    base = os.path.basename(path)
    m = re.search(r"instance_(\d+)_autocorr.dat", base)
    if not m:
        continue
    seed_idx = int(m.group(1))
    by_arm.setdefault(env, {})[seed_idx] = path

print(f"Found arms in engine output: {sorted(by_arm.keys())}")

# Compute SHAs and check against oracle per arm
def sha256_of(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        h.update(f.read())
    return h.hexdigest()

any_arm_passed = False
for arm, seeds in by_arm.items():
    print(f"\n--- arm: {arm} ---")
    all_match = True
    for seed_idx in sorted(oracle_sha):
        eng = seeds.get(seed_idx)
        if eng is None:
            print(f"  seed {seed_idx:02d}: MISSING engine output for this arm")
            all_match = False
            continue
        engine = sha256_of(eng)
        want = oracle_sha[seed_idx]
        ok = engine == want
        print(f"  seed {seed_idx:02d}: engine={engine[:16]}.. oracle={want[:16]}.. {'OK' if ok else 'MISMATCH'}")
        if not ok:
            all_match = False
    if all_match:
        print(f"  >>> arm {arm} matches oracle (W1 byte-match PASSED on this arm)")
        any_arm_passed = True

if not any_arm_passed:
    print("\nW1 CANARY ACCEPTANCE: FAILED — no arm matches the oracle SHAs")
    sys.exit(1)

print("\nW1 CANARY ACCEPTANCE: ALL CHECKS PASSED")
sys.exit(0)
EOF

EXIT_CODE=$?
echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"
exit ${EXIT_CODE}
