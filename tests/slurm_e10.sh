#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e10_validation
#SBATCH --partition=standard
#SBATCH --time=00:01:30
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/e10_validation.o%j
#SBATCH --error=slurm_logs/e10_validation.e%j
#
# E10 — Validation + FiQCI Examples (v1.1.0rc1 Gate)
# RED-SPEC-002 §16
#
# End-to-end validation of the complete Phase E pipeline:
#   VE14: FiQCI GHZ 3q BYO → placement → twin sim → HDF5
#   VE24: Topology columns in production sweep
#   VE25: Multi-calibration (real + synthetic) sweep
#   + QPY round-trip, full pipeline HDF5→Parquet, Bell/Star physics,
#     synthetic cal integration, regression E1–E9
#
# Usage: sbatch tests/slurm_e10.sh
# Expected: E10 VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E10 Validation + FiQCI Examples (v1.1.0rc1 Gate) ==="
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
    python3 "${HPCQC_ROOT}/tests/e10_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
