#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e8_sweep_export
#SBATCH --partition=standard
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/e8_sweep_export.o%j
#SBATCH --error=slurm_logs/e8_sweep_export.e%j
#
# E8 — Sweep Export (HDF5 → 61-Column Parquet) Validation
# RED-SPEC-002 §9, RED-DIRECTIVE-E4-SCHEMA-v1.0 §4
#
# Generates a small sweep via E7, exports to Parquet, validates:
#   VE16: No raw histograms in Parquet
#   VE17: Calibration columns populated
#   VE23: Topology columns populated
#   + schema, round-trip, derived features, CSV
#
# Usage: sbatch tests/slurm_e8.sh
# Expected: E8 VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E8 Sweep Export Validation ==="
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
    python3 "${HPCQC_ROOT}/tests/e8_export_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
