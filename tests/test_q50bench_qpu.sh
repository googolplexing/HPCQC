#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# Q50 Benchmark — Real Q50 quantum computer via FiQCI middleware
#
# This script uses the FiQCI Python environment (NOT the Singularity container).
# The fiqci-vtt-qiskit module provides qiskit-iqm which connects to Q50.
# Q50_CORTEX_URL is set automatically by the module inside q_fiqci partition.
# No API token required — SLURM account is used for billing.
#
# Usage: source env.sh && sbatch tests/test_q50bench_qpu.sh <config>
# Example: sbatch tests/test_q50bench_qpu.sh configs/q50bench_tfim_4q_qpu.yaml
#
#SBATCH --job-name=q50b_qpu
#SBATCH --partition=q_fiqci
#SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --output=slurm_logs/q50bench_qpu.o%j
#SBATCH --error=slurm_logs/q50bench_qpu.e%j

# Source env.sh for HPCQC_ROOT (container/wrapper vars not used here)
source "${SLURM_SUBMIT_DIR}/env.sh"

CONFIG="${1:-configs/q50bench_tfim_4q_qpu.yaml}"

mkdir -p "${HPCQC_ROOT}/slurm_logs" "${HPCQC_ROOT}/results"

SLURM_START_EPOCH=$(date +%s)
echo "=== Q50 Benchmark: Real QPU ==="
echo "Config: ${CONFIG}"
echo "Job ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo ""

# Load FiQCI quantum environment
module use /appl/local/quantum/modulefiles
module --ignore_cache load fiqci-vtt-qiskit

# Select Q50 device
export DEVICES=("Q50")

# Run inside FiQCI environment
# source $RUN_SETUP configures the connection to Q50
# PYTHONPATH includes our framework src/ so imports work
export PROJECT_DIR="${HPCQC_ROOT}"
export CONFIG_PATH="${HPCQC_ROOT}/${CONFIG}"
export PYTHONPATH="${HPCQC_ROOT}/src:${PYTHONPATH}"
export SLURM_START_EPOCH

srun bash -c "source \$RUN_SETUP && python ${HPCQC_ROOT}/tests/vqe_runner.py"

SLURM_END_EPOCH=$(date +%s)
echo ""
echo "Completed: $(date -Iseconds)"
echo "Total SLURM wall time: $(( SLURM_END_EPOCH - SLURM_START_EPOCH )) seconds"
