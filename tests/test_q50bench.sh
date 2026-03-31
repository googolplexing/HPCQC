#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# Q50 Benchmark — Noiseless Aer GPU simulation
# Usage: source env.sh && sbatch tests/test_q50bench_noiseless.sh <config>
# Example: sbatch tests/test_q50bench_noiseless.sh configs/q50bench_tfim_4q_noiseless.yaml
#SBATCH --job-name=q50b_noiseless
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/q50bench_noiseless.o%j
#SBATCH --error=slurm_logs/q50bench_noiseless.e%j

source "${SLURM_SUBMIT_DIR}/env.sh"

# Config can be passed as argument or defaulted
CONFIG="${1:-configs/q50bench_tfim_4q_noiseless.yaml}"

mkdir -p "${HPCQC_ROOT}/slurm_logs" "${HPCQC_ROOT}/results"

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_CONFIG_PATH="${HPCQC_ROOT}/${CONFIG}"

SLURM_START_EPOCH=$(date +%s)
echo "=== Q50 Benchmark: Noiseless ==="
echo "Config: ${CONFIG}"
echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}"
echo "Date: $(date)"
echo ""
export SINGULARITYENV_SLURM_START_EPOCH=$SLURM_START_EPOCH

mask=$HPCQC_GPU_MASK
srun --cpu-bind=$mask ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
  python ${HPCQC_ROOT}/tests/vqe_runner.py

SLURM_END_EPOCH=$(date +%s)
echo ""
echo "Completed: $(date -Iseconds)"
echo "Total SLURM wall time: $(( SLURM_END_EPOCH - SLURM_START_EPOCH )) seconds"
