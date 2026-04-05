#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e9_synthetic_cal
#SBATCH --partition=standard
#SBATCH --time=00:00:25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/e9_synthetic_cal.o%j
#SBATCH --error=slurm_logs/e9_synthetic_cal.e%j
#
# E9 — Synthetic Calibration Tools Validation
# RED-SPEC-002 §10
#
# Tests all perturbation types, physical constraints, provenance,
# batch generation, and twin simulator integration.
#
# Usage: sbatch tests/slurm_e9.sh
# Expected: E9 VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E9 Synthetic Calibration Tools Validation ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Partition: standard (CPU only)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Container: ${HPCQC_CPU_CONTAINER}"
echo "Started:   $(date)"
echo ""

export SINGULARITYENV_PROJECT_DIR=${HPCQC_ROOT}
export SINGULARITYENV_PYTHONPATH=${HPCQC_ROOT}/src

srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/e9_synthetic_cal_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
