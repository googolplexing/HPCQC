#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# NOTE: source env.sh before calling sbatch to set SLURM account
# Usage: source env.sh && sbatch test_checkpoint_resume.sh
#SBATCH --job-name=hpcqc_checkpoint
#SBATCH --partition=standard-g
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/checkpoint_resume.o%j
#SBATCH --error=slurm_logs/checkpoint_resume.e%j

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${HPCQC_ROOT}/slurm_logs"

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"

SLURM_START_EPOCH=$(date +%s)
echo "=== lumi-hpc-qc: Checkpoint/Resume Test ==="
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Date: $(date)"
echo ""
export SINGULARITYENV_SLURM_START_EPOCH=$SLURM_START_EPOCH

srun --cpu-bind=${HPCQC_GPU_MASK} \
  ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
  python ${HPCQC_ROOT}/tests/checkpoint_resume_runner.py

SLURM_END_EPOCH=$(date +%s)
echo ""
echo "Node execution finished: $(date -Iseconds)"
echo "Total SLURM wall time: $(( SLURM_END_EPOCH - SLURM_START_EPOCH )) seconds"
