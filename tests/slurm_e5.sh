#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e5_byo_eval
#SBATCH --partition=standard
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/e5_byo_eval.o%j
#SBATCH --error=slurm_logs/e5_byo_eval.e%j
#
# E5 — BYO Circuit Ingestion + Evaluation-Only Mode
# RED-SPEC-002 §7 — VE15
#
# Tests circuit loading (QPY, QASM, script), connectivity extraction,
# eval-only execution, and parameterization detection.
#
# Runs on standard partition (CPU only). All circuits are ≤4 qubits.
#
# Usage: sbatch tests/slurm_e5.sh
# Expected: E5 VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E5 BYO Circuit + Eval-Only Validation ==="
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
    python3 "${HPCQC_ROOT}/tests/e5_byo_eval_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
