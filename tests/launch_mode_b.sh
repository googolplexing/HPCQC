#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# Launch a Mode B automated experiment
#
# Usage:
#   ./tests/launch_mode_b.sh configs/byo_tfim_8q.yaml
#   ./tests/launch_mode_b.sh configs/heisenberg_3x4_12q.yaml --walltime 02:00:00
#   ./tests/launch_mode_b.sh configs/fermi_hubbard_2x3_12q.yaml --retries 5

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <config.yaml> [--walltime HH:MM:SS] [--retries N] [--poll N]"
    echo ""
    echo "Submits a Mode B controller job on a minimal CPU node."
    echo "The controller submits/monitors GPU child jobs and auto-retries on failure."
    echo ""
    echo "Container/wrapper paths are read from env.sh — edit that file to change them."
    exit 1
fi

CONFIG_YAML="$1"; shift

# Source env.sh for project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

# Defaults
CHILD_WALLTIME="01:00:00"
MAX_RETRIES=3
POLL_INTERVAL=30

while [ $# -gt 0 ]; do
    case "$1" in
        --walltime)  CHILD_WALLTIME="$2"; shift 2 ;;
        --retries)   MAX_RETRIES="$2"; shift 2 ;;
        --poll)      POLL_INTERVAL="$2"; shift 2 ;;
        *)           echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== Mode B Launch ==="
echo "  Config: ${CONFIG_YAML}"
echo "  Project: ${HPCQC_ROOT}"
echo "  Child walltime: ${CHILD_WALLTIME}"
echo "  Max retries: ${MAX_RETRIES}"
echo "  Container: ${HPCQC_GPU_CONTAINER}"
echo ""

export MODEB_CONFIG="${CONFIG_YAML}"
export MODEB_PROJECT_DIR="${HPCQC_ROOT}"
export MODEB_MAX_RETRIES="${MAX_RETRIES}"
export MODEB_POLL_INTERVAL="${POLL_INTERVAL}"
export MODEB_CHILD_WALLTIME="${CHILD_WALLTIME}"

JOB_ID=$(sbatch --export=ALL "${HPCQC_ROOT}/tests/mode_b_controller.sh" 2>&1 | grep -oP '\d+$')

echo "Controller submitted: job ${JOB_ID}"
echo ""
echo "Monitor:"
echo "  squeue --me"
echo "  tail -f slurm_logs/modeb_ctrl.o${JOB_ID}"
echo ""
echo "Results:"
echo "  ./tests/vqa_summary.sh slurm_logs/modeb_*.o*"
