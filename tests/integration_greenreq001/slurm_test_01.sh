#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# GREEN-REQ-001 Integration Test 1: Package imports & version checks
# This test does NOT need GPU but runs inside the container.
# Usage: source env.sh && sbatch tests/integration_greenreq001/slurm_test_01.sh
#SBATCH --job-name=greenreq001_t01_packages
#SBATCH --partition=standard-g
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/greenreq001_t01_packages.o%j
#SBATCH --error=slurm_logs/greenreq001_t01_packages.e%j

source "${SLURM_SUBMIT_DIR}/env.sh"
mkdir -p "${HPCQC_ROOT}/slurm_logs"

echo "=========================================="
echo "GREEN-REQ-001 Integration Test 1"
echo "Package Imports & Version Checks"
echo "=========================================="
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Date: $(date -Iseconds)"
echo "Container: ${HPCQC_GPU_CONTAINER}"
echo ""

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONHASHSEED=0

srun --cpu-bind=${HPCQC_GPU_MASK} \
  ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
  python ${HPCQC_ROOT}/tests/integration_greenreq001/test_01_packages_and_versions.py

EXIT_CODE=$?
echo ""
echo "Exit code: ${EXIT_CODE}"
echo "Completed: $(date -Iseconds)"
exit ${EXIT_CODE}
