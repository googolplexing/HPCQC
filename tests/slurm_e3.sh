#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e3_hdf5_writer
#SBATCH --partition=standard-g
#SBATCH --time=00:00:15
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/e3_hdf5_writer.o%j
#SBATCH --error=slurm_logs/e3_hdf5_writer.e%j
#
# E3 Validation — HDF5-First Writer with WAL
# RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0 System 5
#
# Tests HDF5 write cycle, WAL crash recovery, SWMR mode on Lustre,
# soft link deduplication, measurement stats embedding, debug JSON.
#
# Usage: sbatch tests/slurm_e3.sh
# Expected: E3 VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E3 HDF5 Writer Validation ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_GPU_CONTAINER}"
echo "Filesystem: $(stat -f -c '%T' /flash/ 2>/dev/null || echo 'unknown')"
echo "Started:   $(date)"
echo ""

export SINGULARITYENV_PROJECT_DIR=${HPCQC_ROOT}
export SINGULARITYENV_PYTHONPATH=${HPCQC_ROOT}/src

srun --cpu-bind=${HPCQC_GPU_MASK} ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/e3_hdf5_writer_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
