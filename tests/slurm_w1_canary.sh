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
# Path discipline:
#   - The canary YAML uses repo-relative paths (circuit_script:
#     examples/byo/..., calibrations: examples/q50_..., disorder file:
#     examples/byo/...). The engine resolves these CWD-relative, NOT
#     relative to the YAML's location. So this script `cd`s to
#     $SLURM_SUBMIT_DIR (= $HPCQC_ROOT for repo-root submissions, the
#     project's convention).
#   - The canary YAML pins `output_dir: sweep_output/w1_canary`. The engine
#     writes EVERYTHING (per-instance dats, HDF5, campaign manifest, sweep
#     timing) under that path. The verifier reads from exactly that path —
#     no globbing across unrelated sweep_output dirs, no mtime-based
#     discovery, no risk of stale prior-run files showing up.
#   - Pre-run `rm -rf "${OUTPUT_DIR}"` guarantees a deterministic workdir.
#     This is essential — otherwise the engine's campaign_manifest resume
#     would skip the canary seeds if a prior partial run left a manifest.
#
# Invocation idiom: srun ... python3 path/to/script.py (no -c, no heredoc;
# the wrapper between srun and the singularity container is fragile with
# multi-line args — see LUMI job 18906585's SyntaxError).
#
# Usage: sbatch tests/slurm_w1_canary.sh
# Expected: "W1 CANARY ACCEPTANCE: ALL CHECKS PASSED" on the last line and
# exit code 0.

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

CANARY_YAML="${HPCQC_ROOT}/examples/byo/floquet_dtc_q10_canary_2seed.yaml"
ORACLE="${HPCQC_ROOT}/evidence/W1/gate2_canary/sha256_oracle.txt"
VERIFIER="${HPCQC_ROOT}/tests/_w1_canary_verify.py"
# Engine writes here because the canary YAML pins output_dir to this path.
# This MUST match the YAML's `output_dir:` value (CWD-relative from
# $SLURM_SUBMIT_DIR).
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/sweep_output/w1_canary"

echo "=== W1 Canary — 2-seed byte-match against in-tree oracle ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_CPU_CONTAINER}"
echo "YAML:      ${CANARY_YAML}"
echo "Oracle:    ${ORACLE}"
echo "OutputDir: ${OUTPUT_DIR}"
echo "Verifier:  ${VERIFIER}"
echo "Started:   $(date)"
echo ""

# Pre-run determinism: wipe any prior canary output so the verifier sees only
# files produced by THIS sweep, AND the engine starts without a stale
# campaign manifest that would trigger resume-mode (and skip seeds).
rm -rf "${OUTPUT_DIR}"

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"

# ── (1) Sweep run from the repo root (SLURM_SUBMIT_DIR), so the YAML's
#    repo-relative paths (examples/byo/..., examples/q50_...) resolve.
#    Engine writes to $SLURM_SUBMIT_DIR/sweep_output/w1_canary/ because the
#    YAML pins output_dir to "sweep_output/w1_canary". ──
cd "${SLURM_SUBMIT_DIR}"
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 -m lumi_hpc_qc.sweep.run_sweep "${CANARY_YAML}"

echo ""
echo "=== Byte-match verification ==="

# ── (2) Verify SHAs against the oracle. The verifier walks OUTPUT_DIR
#    recursively for instance_NN_autocorr.dat files, groups by arm
#    (noiseless / device_calibrated path segment), computes SHA256, and
#    PASSES iff at least one arm matches both oracle SHAs (the oracle pins
#    one arm of the 2-seed corpus; the other arm lands with W1.6). ──
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 "${VERIFIER}" \
        --workdir "${OUTPUT_DIR}" \
        --oracle "${ORACLE}"

# set -e ensures we only get here on success (verifier exit 0).
echo ""
echo "Finished: $(date)"
