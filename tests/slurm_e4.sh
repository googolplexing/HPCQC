#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e4_twin_sim
#SBATCH --partition=standard
#SBATCH --time=00:00:25
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/e4_twin_sim.o%j
#SBATCH --error=slurm_logs/e4_twin_sim.e%j
#
# E4 — Multi-Calibration Twin Simulation Battery
# RED-SPEC-002 §4
#
# Tests all 11 noise environments per placement, noiseless deduplication,
# and calibration sensitivity. All 4q circuits — CPU is optimal.
#
# Usage: sbatch tests/slurm_e4.sh
# Expected: E4 VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E4 Twin Simulation Battery Validation ==="
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
    python3 "${HPCQC_ROOT}/tests/e4_twin_sim_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
