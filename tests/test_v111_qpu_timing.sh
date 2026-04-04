#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# QPU Timing Benchmark — Measures per-circuit and per-batch overhead on Q50
#
# RED-RESP-PACKING-v1.0 §2.7 — prerequisite for all QPU cost estimates.
#
# Submits:
#   1. Batch of 10 circuits (SU2 reps=2, 4q star, 4096 shots)
#   2. Batch of 1 circuit (same)
#   3. Computes: per_circuit_ms, per_batch_overhead_ms, inferred reset method
#
# Uses FiQCI module (NOT the Singularity container).
#
# Usage:
#   source env.sh
#   sbatch tests/test_v111_qpu_timing.sh
#
#SBATCH --job-name=qpu_timing
#SBATCH --partition=q_fiqci
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm_logs/qpu_timing.o%j
#SBATCH --error=slurm_logs/qpu_timing.e%j

source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p "${HPCQC_ROOT}/slurm_logs" "${HPCQC_ROOT}/results"

echo "═══════════════════════════════════════════════════════════"
echo "  QPU TIMING BENCHMARK — VTT Q50"
echo "  RED-RESP-PACKING-v1.0 §2.7"
echo "  Job ${SLURM_JOB_ID}"
echo "  Date: $(date -Iseconds)"
echo "═══════════════════════════════════════════════════════════"
echo ""

# Load FiQCI quantum environment
module use /appl/local/quantum/modulefiles
module --ignore_cache load fiqci-vtt-qiskit
export DEVICES=("Q50")

# Run the benchmark inside FiQCI environment
srun --account ${HPCQC_ACCOUNT} bash -c "source \$RUN_SETUP && python -u ${HPCQC_ROOT}/tests/qpu_timing_benchmark.py"

echo ""
echo "Completed: $(date -Iseconds)"
