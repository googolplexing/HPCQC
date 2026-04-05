#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e1_placement
#SBATCH --partition=standard-g
#SBATCH --time=00:00:15
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/e1_placement.o%j
#SBATCH --error=slurm_logs/e1_placement.e%j
#
# E1 Validation — General Placement Solver
# RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0 System 1
#
# Tests VF2 subgraph isomorphism on Q50, calibration adapter,
# multi-round packing, topology equivalence, scoring strategies.
#
# Usage: sbatch tests/slurm_e1.sh
# Expected: E1 VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E1 Placement Solver Validation ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_GPU_CONTAINER}"
echo "Started:   $(date)"
echo ""

export SINGULARITYENV_PROJECT_DIR=${HPCQC_ROOT}
export SINGULARITYENV_PYTHONPATH=${HPCQC_ROOT}/src

srun --cpu-bind=${HPCQC_GPU_MASK} ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/e1_placement_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
