#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# NOTE: source env.sh before calling sbatch to set SLURM account
# Usage: source env.sh && sbatch tests/test_byo_tfim_8q_mps.sh
#SBATCH --job-name=hpcqc_byo_mps
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/byo_tfim_8q_mps.o%j
#SBATCH --error=slurm_logs/byo_tfim_8q_mps.e%j

source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p "${HPCQC_ROOT}/slurm_logs" "${HPCQC_ROOT}/results"

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_CONFIG_PATH="${HPCQC_ROOT}/configs/byo_tfim_8q_mps.yaml"

SLURM_START_EPOCH=$(date +%s)
echo "=== lumi-hpc-qc: BYO TFIM 8q MPS (CPU-only) ==="
echo "Job ${SLURM_JOB_ID} on ${SLURM_NNODES} node(s)"
echo "Node list: ${SLURM_NODELIST}"
echo "Date: $(date)"
echo ""
export SINGULARITYENV_SLURM_START_EPOCH=$SLURM_START_EPOCH

srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
  python ${HPCQC_ROOT}/tests/vqe_runner.py

SLURM_END_EPOCH=$(date +%s)
echo ""
echo "Node execution finished: $(date -Iseconds)"
echo "Total SLURM wall time: $(( SLURM_END_EPOCH - SLURM_START_EPOCH )) seconds"
