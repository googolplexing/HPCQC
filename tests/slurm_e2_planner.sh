#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e2_planner
#SBATCH --partition=standard
#SBATCH --time=00:00:20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/e2_planner.o%j
#SBATCH --error=slurm_logs/e2_planner.e%j
#
# E2 Tiered Execution Planner Validation
# RED-SPEC-002 §5
# VE10: 4q→aer_cpu, 12q→aer_gpu routing
#
# Tests routing logic + CPU parallel dispatch. Only 16 CPUs needed
# (test uses 16 parallel workers, not full 128).
#
# Usage: sbatch tests/slurm_e2_planner.sh
# Expected: E2 EXECUTION PLANNER: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E2 Tiered Execution Planner Validation ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Partition: standard (CPU only)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Container: ${HPCQC_CPU_CONTAINER}"
echo "Started:   $(date)"
echo ""

export SINGULARITYENV_PROJECT_DIR=${HPCQC_ROOT}
export SINGULARITYENV_PYTHONPATH=${HPCQC_ROOT}/src
export SINGULARITYENV_OMP_NUM_THREADS=1

srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/e2_execution_planner_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
