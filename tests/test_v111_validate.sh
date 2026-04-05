#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# v1.1.1 Validation — Packing + Batch + DSatur + VE19 Tests
#
# Runs inside the Singularity container on a CPU node.
# Tests: Test A (batch ordering), Test B (multi-batch boundary),
#        Test C (DSatur optimality), Test D (non-overlap),
#        VE19 amendment (YAML interval override)
#
# Usage:
#   source env.sh
#   sbatch tests/test_v111_validate.sh
#
# Expected output: ALL CHECKS PASSED for both test suites
#
#SBATCH --job-name=v111_validate
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0G
#SBATCH --output=slurm_logs/v111_validate.o%j
#SBATCH --error=slurm_logs/v111_validate.e%j

source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p "${HPCQC_ROOT}/slurm_logs"

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"

# CRITICAL: Disable C++ thread pools (BLAS/OpenMP) before Python starts.
# Parallelism comes from multiprocessing.Pool, not per-worker BLAS.
# Without this, fork() copies locked mutexes → child process deadlock.
export OMP_NUM_THREADS=1
# Pass into Singularity container
export SINGULARITYENV_OMP_NUM_THREADS=1

SLURM_START_EPOCH=$(date +%s)
echo "═══════════════════════════════════════════════════════════"
echo "  v1.1.1 VALIDATION SUITE"
echo "  Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo "  Date: $(date -Iseconds)"
echo "═══════════════════════════════════════════════════════════"
echo ""

FAIL=0

# ── Test 1: Packing + Batch ordering tests (Test A/B/C/D) ──
echo "┌─────────────────────────────────────────────────────────┐"
echo "│ TEST SUITE 1: Packing + Batch (Test A/B/C/D)           │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
  python ${HPCQC_ROOT}/tests/test_v111_packing_batch.py

if [ $? -ne 0 ]; then
    echo ""
    echo "  *** TEST SUITE 1 FAILED — Test A or Test B are MERGE-BLOCKING ***"
    FAIL=1
fi

echo ""
echo ""

# ── Test 2: E7 sweep validation (includes VE19 amendment) ──
echo "┌─────────────────────────────────────────────────────────┐"
echo "│ TEST SUITE 2: E7 Sweep Validation (includes VE19)      │"
echo "└─────────────────────────────────────────────────────────┘"
echo ""

srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
  python ${HPCQC_ROOT}/tests/e7_sweep_validation.py

if [ $? -ne 0 ]; then
    echo ""
    echo "  *** TEST SUITE 2 FAILED ***"
    FAIL=1
fi

echo ""
SLURM_END_EPOCH=$(date +%s)
ELAPSED=$(( SLURM_END_EPOCH - SLURM_START_EPOCH ))

echo "═══════════════════════════════════════════════════════════"
echo "  v1.1.1 VALIDATION COMPLETE"
echo "  Wall time: ${ELAPSED} seconds"
if [ $FAIL -eq 0 ]; then
    echo "  STATUS: ALL SUITES PASSED ✓"
    echo "  → Safe to merge per RED-RESP-PACKING-v1.0 merge order"
else
    echo "  STATUS: FAILURES DETECTED ✗"
    echo "  → DO NOT MERGE — fix failures first"
fi
echo "═══════════════════════════════════════════════════════════"

exit $FAIL
