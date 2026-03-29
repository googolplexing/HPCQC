#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# GREEN-REQ-001 Integration Test 3: Checkpoint/Resume + PYTHONHASHSEED
# Requires GPU node for Aer statevector VQE
# Usage: source env.sh && sbatch tests/integration_greenreq001/slurm_test_03.sh
#SBATCH --job-name=greenreq001_t03_checkpoint
#SBATCH --partition=standard-g
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/greenreq001_t03_checkpoint.o%j
#SBATCH --error=slurm_logs/greenreq001_t03_checkpoint.e%j

source "${SLURM_SUBMIT_DIR}/env.sh"
mkdir -p "${HPCQC_ROOT}/slurm_logs"

echo "=========================================="
echo "GREEN-REQ-001 Integration Test 3"
echo "Checkpoint/Resume + Determinism"
echo "=========================================="
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Date: $(date -Iseconds)"
echo "Container: ${HPCQC_GPU_CONTAINER}"
echo ""

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONHASHSEED=0

SLURM_START_EPOCH=$(date +%s)
export SINGULARITYENV_SLURM_START_EPOCH=$SLURM_START_EPOCH

srun --cpu-bind=${HPCQC_GPU_MASK} \
  ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
  python ${HPCQC_ROOT}/tests/integration_greenreq001/test_03_checkpoint_and_determinism.py

EXIT_CODE=$?
SLURM_END_EPOCH=$(date +%s)
echo ""
echo "Exit code: ${EXIT_CODE}"
echo "Wall time: $(( SLURM_END_EPOCH - SLURM_START_EPOCH )) seconds"
echo "Completed: $(date -Iseconds)"
exit ${EXIT_CODE}
