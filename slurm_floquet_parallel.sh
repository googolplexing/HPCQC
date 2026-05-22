#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Run 10 Floquet gate instances in parallel on ONE LUMI standard node.
# Matches the HPCQC standard-partition pattern (tests/slurm_e2_stress.sh):
# single srun, single container instance, multiprocessing.Pool inside Python.
#
# Usage:
#   sbatch slurm_floquet_parallel.sh noiseless
#   sbatch slurm_floquet_parallel.sh q50-noise /full/path/to/calibration.json
#
#SBATCH --job-name=floquet_par
#SBATCH --account=project_462000055
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/floquet_par.o%j
#SBATCH --error=slurm_logs/floquet_par.e%j

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

# ─────────────── args ───────────────
MODE="${1:-}"
CAL_PATH="${2:-}"

if [[ "$MODE" != "noiseless" && "$MODE" != "q50-noise" ]]; then
    echo "ERROR: first arg must be 'noiseless' or 'q50-noise', got '${MODE}'"
    echo "Usage: sbatch $0 noiseless"
    echo "       sbatch $0 q50-noise /path/to/calibration.json"
    exit 2
fi
if [[ "$MODE" == "q50-noise" ]]; then
    if [[ -z "$CAL_PATH" ]]; then
        echo "ERROR: q50-noise mode needs a calibration JSON as 2nd arg"
        exit 2
    fi
    if [[ ! -f "$CAL_PATH" ]]; then
        echo "ERROR: calibration file not found: $CAL_PATH"
        exit 2
    fi
fi

# ─────────────── output dir ───────────────
mkdir -p "${HPCQC_ROOT}/slurm_logs"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${HPCQC_ROOT}/results/floquet_${MODE//-/_}_${TS}_job${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "=== Floquet 10-instance parallel run (${MODE}) ==="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "CPUs/task : ${SLURM_CPUS_PER_TASK}"
echo "Container : ${HPCQC_CPU_CONTAINER}"
echo "Wrapper   : ${HPCQC_CPU_WRAPPER}"
echo "HPCQC root: ${HPCQC_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "Calib.    : ${CAL_PATH:-(noiseless, not used)}"
echo "Started   : $(date)"
echo

# ─────────────── env propagated into the container ───────────────
export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"

# Suppress ROCm/HSA GPU init on the standard partition (matches
# tests/slurm_e2_stress.sh).
export SINGULARITYENV_ROCR_VISIBLE_DEVICES=""
export SINGULARITYENV_HSA_TOOLS_LIB=""

# Keep qiskit's parallel_map in SERIAL mode inside the multiprocessing
# Pool. Pool workers are daemonic; daemonic processes can't spawn
# children, so any nested parallel_map call would raise
# AssertionError("daemonic processes are not allowed to have children").
# The runner also passes num_processes=1 to transpile() explicitly, but
# this catches any other call sites inside qiskit/qiskit-aer.
export SINGULARITYENV_QISKIT_IN_PARALLEL=TRUE

# ─────────────── calibration CLI arg ───────────────
CAL_ARG=""
if [[ "$MODE" == "q50-noise" ]]; then
    CAL_ARG="--calibration ${CAL_PATH}"
fi

# ─────────────── single srun → single container → Pool inside ───────────────
srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/floquet_runner.py" \
        --backend "${MODE}" \
        ${CAL_ARG} \
        --output-dir "${OUTDIR}" \
        --num-instances 10

EXIT_CODE=$?

echo
echo "=== Finished: $(date)  exit=${EXIT_CODE} ==="
echo
echo "Aggregate + plot:"
echo "  srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \\"
echo "      python3 ${HPCQC_ROOT}/aggregate_floquet.py ${OUTDIR}"

exit ${EXIT_CODE}
