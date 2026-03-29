#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# GREEN-REQ-001 Integration Test Suite — Master Launcher
#
# Submits all 3 integration tests as separate SLURM jobs.
# Test 1 runs independently. Tests 2 and 3 can run in parallel.
#
# Usage:
#   cd ~/HPCQC                          # project root on LUMI
#   source env.sh                        # load container paths + SLURM account
#   ./tests/integration_greenreq001/launch_all.sh
#
# Monitor:
#   squeue --me
#
# Results:
#   cat slurm_logs/greenreq001_t01_packages.o<JOBID>
#   cat slurm_logs/greenreq001_t02_aer_vqe.o<JOBID>
#   cat slurm_logs/greenreq001_t03_checkpoint.o<JOBID>

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Source env.sh from project root
source "${PROJECT_ROOT}/env.sh"

echo "=========================================================="
echo "  GREEN-REQ-001 — Container Integration Test Suite"
echo "=========================================================="
echo "  Date:      $(date -Iseconds)"
echo "  Project:   ${HPCQC_ROOT}"
echo "  Container: ${HPCQC_GPU_CONTAINER}"
echo "  Account:   ${HPCQC_ACCOUNT}"
echo "=========================================================="
echo ""

# Verify container exists
if [ ! -f "${HPCQC_GPU_CONTAINER}" ]; then
    echo "ERROR: Container not found at ${HPCQC_GPU_CONTAINER}"
    echo ""
    echo "If the rebuilt container is at a different path, update env.sh:"
    echo "  export HPCQC_GPU_CONTAINER=/path/to/new/07-patchelf-fix.sif"
    exit 1
fi

# Verify test scripts exist
for script in slurm_test_01.sh slurm_test_02.sh slurm_test_03.sh; do
    if [ ! -f "${SCRIPT_DIR}/${script}" ]; then
        echo "ERROR: Test script not found: ${SCRIPT_DIR}/${script}"
        exit 1
    fi
done

mkdir -p "${HPCQC_ROOT}/slurm_logs"

# Submit all tests
echo "Submitting integration tests..."
echo ""

JOB1=$(cd "${HPCQC_ROOT}" && sbatch --export=ALL "${SCRIPT_DIR}/slurm_test_01.sh" 2>&1 | grep -oP '\d+$')
echo "  Test 1 (Packages & Versions):     job ${JOB1:-FAILED}"

JOB2=$(cd "${HPCQC_ROOT}" && sbatch --export=ALL "${SCRIPT_DIR}/slurm_test_02.sh" 2>&1 | grep -oP '\d+$')
echo "  Test 2 (Aer GPU + VQE):           job ${JOB2:-FAILED}"

JOB3=$(cd "${HPCQC_ROOT}" && sbatch --export=ALL "${SCRIPT_DIR}/slurm_test_03.sh" 2>&1 | grep -oP '\d+$')
echo "  Test 3 (Checkpoint + Determinism): job ${JOB3:-FAILED}"

echo ""
echo "=========================================================="
echo "  3 jobs submitted"
echo "=========================================================="
echo ""
echo "  Monitor progress:"
echo "    squeue --me"
echo ""
echo "  View results when complete:"
echo "    cat slurm_logs/greenreq001_t01_packages.o${JOB1}"
echo "    cat slurm_logs/greenreq001_t02_aer_vqe.o${JOB2}"
echo "    cat slurm_logs/greenreq001_t03_checkpoint.o${JOB3}"
echo ""
echo "  Quick pass/fail summary:"
echo "    grep -E 'PASS|FAIL|Exit code' slurm_logs/greenreq001_*.o*"
echo ""
echo "  Expected wall times:"
echo "    Test 1: ~1 minute  (import checks only)"
echo "    Test 2: ~5–10 minutes (VQE convergence with 200 iters)"
echo "    Test 3: ~5–10 minutes (checkpoint + resume VQE)"
echo ""
echo "  Report results to Team Green within 48 hours."
