#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# NOTE: source env.sh before calling sbatch to set SLURM account
# Usage: source env.sh && sbatch test_smoke.sh
#SBATCH --job-name=hpcqc_smoke_test
#SBATCH --partition=standard-g
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=slurm_logs/smoke_test.o%j
#SBATCH --error=slurm_logs/smoke_test.e%j

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

mkdir -p "${HPCQC_ROOT}/slurm_logs"

echo "=== lumi-hpc-qc Smoke Test ==="
echo "Job ${SLURM_JOB_ID} on $(hostname)"
echo "Date: $(date)"
echo ""

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"

srun --cpu-bind=${HPCQC_GPU_MASK} \
  ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
  python ${HPCQC_ROOT}/tests/smoke_test_runner.py

echo ""
echo "Smoke test completed: $(date)"
