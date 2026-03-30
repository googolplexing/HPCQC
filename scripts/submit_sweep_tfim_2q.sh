#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Submit 20-seed sweep for TFIM 2q noiseless
# Usage: bash scripts/submit_sweep_tfim_2q.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"

source "${PROJECT_ROOT}/env.sh"

CONFIGS_DIR="${PROJECT_ROOT}/configs/seed_sweep/tfim_2q"

if [ ! -d "${CONFIGS_DIR}" ]; then
    echo "ERROR: Config directory not found: ${CONFIGS_DIR}"
    echo "Run generate_seed_sweep.py first."
    exit 1
fi

echo "=== TFIM 2q Seed Sweep ==="
echo "Date: $(date)"
echo "Configs: ${CONFIGS_DIR}"
echo ""

JOB_COUNT=0

for cfg in ${CONFIGS_DIR}/*.yaml; do
    seed=$(basename "${cfg}" .yaml | grep -oP 'seed\K\d+')

    JOB_ID=$(sbatch --parsable \
        --job-name=t2q_s${seed} \
        --partition=standard-g \
        --time=00:30:00 \
        --nodes=1 \
        --gpus-per-node=8 \
        --ntasks-per-node=1 \
        --cpus-per-task=56 \
        --output=slurm_logs/tfim_2q_seed${seed}.o%j \
        --error=slurm_logs/tfim_2q_seed${seed}.e%j \
        --wrap="export SINGULARITYENV_PROJECT_DIR=${HPCQC_ROOT} && \
        export SINGULARITYENV_PYTHONPATH=${HPCQC_ROOT}/src && \
        srun --cpu-bind=${HPCQC_GPU_MASK} ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
        python ${HPCQC_ROOT}/scripts/run_vqe.py ${cfg}")

    echo "  seed ${seed}: job ${JOB_ID}"
    JOB_COUNT=$((JOB_COUNT + 1))
done

echo ""
echo "Submitted ${JOB_COUNT} jobs for TFIM 2q"
echo "Monitor: squeue --me"
