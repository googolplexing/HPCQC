#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e6b_mixed_packing
#SBATCH --partition=standard
#SBATCH --time=00:00:15
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/e6b_mixed_packing.o%j
#SBATCH --error=slurm_logs/e6b_mixed_packing.e%j
#
# E6b — Mixed-Experiment Packing Validation
# RED-SPEC-002 §15
#
# Tests heterogeneous circuit packing: different circuits from different
# experiments share a single QPU submission.
#   VE20: Two circuits packed, demuxed, results match independent runs
#   + 3-experiment packing, deterministic, edge non-overlap, error isolation
#
# Usage: sbatch tests/slurm_e6b.sh
# Expected: E6b VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E6b Mixed-Experiment Packing Validation ==="
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
    python3 "${HPCQC_ROOT}/tests/e6b_mixed_packing_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
