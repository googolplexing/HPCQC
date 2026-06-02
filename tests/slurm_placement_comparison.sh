#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=placement_comparison
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/placement_comparison.o%j
#SBATCH --error=slurm_logs/placement_comparison.e%j
#
# Placement-strategy comparison diagnostic (not a CI gate).
# Compares fidelity-top-N vs disjoint vs parallel-round packing for a circuit of
# a given SHAPE on one calibration, and emits paste-ready sweep-YAML permutations
# (manual / solver / manual+solver union / diversity). Solver-only (rustworkx +
# cal adapter) — NO aer/h5py.
#
# Usage:
#   sbatch tests/slurm_placement_comparison.sh [CAL_JSON] [N_PLACEMENTS] [CHAIN_QUBITS] [SHAPE]
# Defaults (campaign): cal 08c3c70f, N=12, 10 qubits, SHAPE=chain.
# SHAPE in: chain | ring | star | grid | ladder | complete
# Examples:
#   sbatch tests/slurm_placement_comparison.sh
#   sbatch tests/slurm_placement_comparison.sh examples/q50_calibration_20260601_5c75890f.json 12 10 chain
#   sbatch tests/slurm_placement_comparison.sh examples/q50_calibration_20260524_08c3c70f.json 5 5 star

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"

mkdir -p slurm_logs

CAL="${1:-examples/q50_calibration_20260524_08c3c70f.json}"
N="${2:-12}"
QUBITS="${3:-10}"
SHAPE="${4:-chain}"

echo "=== Placement Strategy Comparison (cal=${CAL} N=${N} qubits=${QUBITS} shape=${SHAPE}) ==="
srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${SLURM_SUBMIT_DIR}/tests/placement_strategy_comparison.py" \
    "${SLURM_SUBMIT_DIR}/${CAL}" "${N}" "${QUBITS}" "${SHAPE}"
