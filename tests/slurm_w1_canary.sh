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
# Two-step layout:
#   (1) Sweep run via the canonical CLI entry point
#       (python3 -m lumi_hpc_qc.sweep.run_sweep), executed inside the
#       container after `cd $WORKDIR` so sweep_output/ lands in the
#       job-isolated workdir.
#   (2) Byte-match verification via tests/_w1_canary_verify.py (a small
#       Python helper, name underscore-prefixed so pytest doesn't try to
#       collect it as a test module).
#
# Both steps use the slurm_e1.sh-style invocation (no `-c` or heredocs;
# the wrapper between srun and the container is fragile with multi-line
# args — see W1 canary job 18906585's SyntaxError on `\nimport sys`).
#
# Usage: sbatch tests/slurm_w1_canary.sh
# Expected: "W1 CANARY ACCEPTANCE: ALL CHECKS PASSED" on the last line and
# exit code 0.

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

CANARY_YAML="${HPCQC_ROOT}/examples/byo/floquet_dtc_q10_canary_2seed.yaml"
ORACLE="${HPCQC_ROOT}/evidence/W1/gate2_canary/sha256_oracle.txt"
WORKDIR="${SLURM_SUBMIT_DIR}/sweep_output_w1_canary_${SLURM_JOB_ID}"
VERIFIER="${HPCQC_ROOT}/tests/_w1_canary_verify.py"

mkdir -p "${WORKDIR}"

echo "=== W1 Canary — 2-seed byte-match against in-tree oracle ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_CPU_CONTAINER}"
echo "YAML:      ${CANARY_YAML}"
echo "Oracle:    ${ORACLE}"
echo "Workdir:   ${WORKDIR}"
echo "Verifier:  ${VERIFIER}"
echo "Started:   $(date)"
echo ""

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"

# ── (1) Run the sweep on the W1 engine. cd into WORKDIR so sweep_output/
#    (the engine's default output root) lands in this job's workdir, isolated
#    from prior runs' sweep_output*/ directories in the submit dir. Uses the
#    canonical CLI entry point — same shape as the gate-2 yaml header. ──
cd "${WORKDIR}"
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 -m lumi_hpc_qc.sweep.run_sweep "${CANARY_YAML}"

echo ""
echo "=== Byte-match verification ==="

# ── (2) Verify SHAs against the oracle. The verifier walks WORKDIR for
#    instance_NN_autocorr.dat files, groups by arm, computes SHA256, and
#    PASSES iff at least one arm matches both oracle SHAs (the oracle pins
#    one arm of the 2-seed corpus; the other lands with W1.6). ──
cd "${SLURM_SUBMIT_DIR}"
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 "${VERIFIER}" \
        --workdir "${WORKDIR}" \
        --oracle "${ORACLE}"

# set -e ensures we only get here on success (verifier exit 0).
echo ""
echo "Finished: $(date)"
