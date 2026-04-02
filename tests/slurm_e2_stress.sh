#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e2_cpu_stress
#SBATCH --partition=standard
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/e2_cpu_stress.o%j
#SBATCH --error=slurm_logs/e2_cpu_stress.e%j
#
# E2.1 Stress Test — CPU Parallelism for Tiered Execution Engine
# RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0 System 4
#
# Runs on standard partition (CPU only, no GPU billing).
# Tests 16, 64, and 128 parallel AerSimulator(density_matrix) instances
# via multiprocessing.Pool. Validates:
#   - No result corruption under parallel execution
#   - Reproducibility (same seed = same energy)
#   - Scaling behavior (speedup vs sequential)
#   - Memory footprint stays within node limits
#
# This test gates the tiered engine's parallelism model. If it fails,
# fallback is subprocess-level isolation via SLURM job arrays.
#
# Usage: sbatch tests/slurm_e2_stress.sh
# Expected: E2.1 STRESS TEST: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E2.1 CPU Parallelism Stress Test ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Partition: standard (CPU only)"
echo "CPUs:      ${SLURM_CPUS_PER_TASK}"
echo "Container: ${HPCQC_CPU_CONTAINER}"
echo "Started:   $(date)"
echo ""

export SINGULARITYENV_PROJECT_DIR=${HPCQC_ROOT}
export SINGULARITYENV_PYTHONPATH=${HPCQC_ROOT}/src

# Use CPU wrapper (no GPU affinity needed)
srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/e2_cpu_stress_test.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
