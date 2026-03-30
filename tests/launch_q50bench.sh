#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# Q50 Benchmark Suite — launch all 7 models for a given execution mode
#
# Usage:
#   source env.sh
#   ./tests/launch_q50bench.sh noiseless              # Aer GPU statevector (ideal)
#   ./tests/launch_q50bench.sh noiseless_controlled    # Aer GPU statevector (noisy optimizer, no noise)
#   ./tests/launch_q50bench.sh noisy                   # Aer GPU with Q50 noise model
#   ./tests/launch_q50bench.sh qpu                     # Real Q50 via FiQCI
#
# Each model is submitted as a separate SLURM job.
# Results can be compared after all jobs complete:
#   ./tests/vqa_summary.sh slurm_logs/q50bench_*.o*

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../env.sh"

MODE="${1:-}"

if [ -z "$MODE" ] || [[ ! "$MODE" =~ ^(noiseless|noiseless_controlled|noisy|qpu)$ ]]; then
    echo "Usage: $0 <noiseless|noiseless_controlled|noisy|qpu>"
    echo ""
    echo "Modes:"
    echo "  noiseless             — Aer GPU statevector, ideal optimizer (L-BFGS-B + param shift)"
    echo "  noiseless_controlled  — Aer GPU statevector, noisy optimizer (SPSA), no hardware noise"
    echo "  noisy                 — Aer GPU density_matrix with Q50 noise model"
    echo "  qpu                   — Real VTT Q50 quantum computer via FiQCI"
    echo ""
    echo "Models (7): TFIM 2q, TFIM 4q, TFIM 8q, H2 4q, QAOA 8q, FH 4q, Heisenberg 4q"
    echo ""
    echo "The 5-point comparison chain (each step changes one variable):"
    echo "  noiseless → noiseless_controlled → topology_noiseless → noisy → qpu"
    echo "  (topology_noiseless mode will be added in Phase B)"
    exit 1
fi

# All 7 benchmark models (Phase A: added fh_4q and heis_4q)
MODELS="tfim_2q tfim_4q tfim_8q h2_4q qaoa_8q fh_4q heis_4q"

# noiseless_controlled uses the same SLURM script as noiseless
# (both run on standard-g with statevector — the difference is in the YAML config:
#  noiseless uses L-BFGS-B + parameter_shift, controlled uses SPSA + no gradient)
case "$MODE" in
    noiseless|noiseless_controlled)
        SCRIPT="tests/test_q50bench_noiseless.sh"
        ;;
    noisy)
        SCRIPT="tests/test_q50bench_noisy.sh"
        ;;
    qpu)
        SCRIPT="tests/test_q50bench_qpu.sh"
        ;;
esac

echo "=== Q50 Benchmark Suite: ${MODE} ==="
echo "Date: $(date)"
echo ""

JOB_IDS=()

for model in $MODELS; do
    config="configs/q50bench_${model}_${MODE}.yaml"
    if [ ! -f "${HPCQC_ROOT}/${config}" ]; then
        echo "  SKIP: ${config} (not found)"
        continue
    fi

    # Submit with config as argument
    # Note: sbatch passes arguments after the script name
    JOB_ID=$(sbatch --export=ALL "${HPCQC_ROOT}/${SCRIPT}" "${config}" 2>&1 | grep -oP '\d+$')

    if [ -n "$JOB_ID" ]; then
        echo "  ${model}: job ${JOB_ID} (${config})"
        JOB_IDS+=("$JOB_ID")
    else
        echo "  ${model}: FAILED to submit"
    fi
done

echo ""
echo "Submitted ${#JOB_IDS[@]} jobs"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor: squeue --me"
echo "Results: ./tests/vqa_summary.sh slurm_logs/q50bench_${MODE}.o*"
echo ""
echo "Note: Models without q50bench configs will show as SKIP."
echo "Missing configs will be generated in Phase B (config generator)."
