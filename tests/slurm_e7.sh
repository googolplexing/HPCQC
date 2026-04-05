#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=e7_sweep_engine
#SBATCH --partition=standard
#SBATCH --time=00:03:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm_logs/e7_sweep_engine.o%j
#SBATCH --error=slurm_logs/e7_sweep_engine.e%j
#
# E7 — Sweep Engine Orchestrator Validation
# RED-SPEC-002 §§1–17
#
# Tests the full YAML → HDF5 sweep pipeline:
#   VE18: Full sweep (TFIM 4q, 1 cal, all placements, all envs, 2 seeds)
#   VE19: Tiered measurement stats intervals (Tier A=5, B=20, full=10)
#   VE22: Multi-topology sweep (chain + star placements)
#
# 30 min allocated — the full sweep across 379+ chain placements × 11 envs
# × 2 seeds takes ~10 min on 16 CPU cores. Plus the multi-topology sweep.
#
# Usage: sbatch tests/slurm_e7.sh
# Expected: E7 VALIDATION: ALL CHECKS PASSED

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== E7 Sweep Engine Orchestrator Validation ==="
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
    python3 "${HPCQC_ROOT}/tests/e7_sweep_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
