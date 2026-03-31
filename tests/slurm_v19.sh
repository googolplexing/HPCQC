#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=v19_validation
#SBATCH --partition=standard-g
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/v19_validation.o%j
#SBATCH --error=slurm_logs/v19_validation.e%j
#
# V19 Acceptance Test — Measurement Stats End-to-End
# RED-RESP-V19-v1.0 §1 Q3: 6-step validation
#
# Runs two short VQE experiments (TFIM 2q, 3 iterations each):
#   1. With capture_measurement_stats=true → verify sidecar + HDF5
#   2. With capture_measurement_stats=false → verify absence
#
# Usage: sbatch tests/slurm_v19.sh
# Expected: V19 VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== V19 Acceptance Test — Measurement Stats ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_GPU_CONTAINER}"
echo "Started:   $(date)"
echo ""

export SINGULARITYENV_PROJECT_DIR=${HPCQC_ROOT}
export SINGULARITYENV_PYTHONPATH=${HPCQC_ROOT}/src

srun --cpu-bind=${HPCQC_GPU_MASK} ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/v19_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
