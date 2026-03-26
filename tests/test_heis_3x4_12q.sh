#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# NOTE: source env.sh before calling sbatch to set SLURM account
# Usage: source env.sh && sbatch test_heis_3x4_12q.sh
#SBATCH --job-name=hpcqc_heis_3x4_12q
#SBATCH --partition=standard-g
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/heis_3x4_12q.o%j
#SBATCH --error=slurm_logs/heis_3x4_12q.e%j

# Source central environment config (container paths, wrappers, etc.)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${HPCQC_ROOT}/slurm_logs" "${HPCQC_ROOT}/results"

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_CONFIG_PATH="${HPCQC_ROOT}/configs/heisenberg_3x4_12q.yaml"

SLURM_START_EPOCH=$(date +%s)
echo "=== lumi-hpc-qc: Heisenberg 3x4 12q (adiabatic) ==="
echo "Job ${SLURM_JOB_ID} on ${SLURM_NNODES} node(s)"
echo "Node list: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo ""
export SINGULARITYENV_SLURM_START_EPOCH=$SLURM_START_EPOCH

srun --cpu-bind=${HPCQC_GPU_MASK} \
  ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
  python ${HPCQC_ROOT}/tests/vqe_runner.py

SLURM_END_EPOCH=$(date +%s)
echo ""
echo "Node execution finished: $(date -Iseconds)"
echo "Total SLURM wall time: $(( SLURM_END_EPOCH - SLURM_START_EPOCH )) seconds"
