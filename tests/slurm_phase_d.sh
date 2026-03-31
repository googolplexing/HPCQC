#!/bin/bash
#SBATCH --job-name=phase_d_val
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/phase_d_val.o%j
#SBATCH --error=slurm_logs/phase_d_val.e%j

# Phase D validation — runs inside the container on LUMI standard-g
# Usage: sbatch tests/slurm_phase_d.sh
#
# Expected runtime: < 2 minutes
# Expected output:  35/35 PASSED in slurm_logs/phase_d_val.o<jobid>
#
# No --cpu-bind: HPCQC_GPU_MASK is an 8-GCD mask for full-node GPU jobs.
# This job requests 1 GPU for container access only — pure Python, no GPU compute.

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== Phase D Validation — LUMI container run ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_GPU_CONTAINER}"
echo "Started:   $(date)"
echo ""

# Run Phase D validation (35 checks — types, schema, quality gate)
srun ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/phase_d_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

# Verify Phase D export dependencies are present in the container
echo ""
echo "--- Container package check ---"
srun ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/check_phase_d_packages.py"

exit ${EXIT_CODE}
