#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Run N Floquet gate instances in parallel on ONE LUMI standard node.
# Matches the HPCQC standard-partition pattern (tests/slurm_e2_stress.sh):
# single srun, single container instance, multiprocessing.Pool inside Python.
#
# Defaults baked into this file: 40 instances, 60 kicks/circuits per
# instance, 1000 shots (both noiseless and q50-noise).
#
# Usage:
#   sbatch slurm_floquet_40i_60k_1000s.sh noiseless
#   sbatch slurm_floquet_40i_60k_1000s.sh q50-noise /full/path/to/calibration.json
#
# Tunable knobs (env vars; defaults shown). Override at submit time, e.g.:
#   FLOQUET_SHOTS=100 sbatch slurm_floquet_40i_60k_1000s.sh noiseless
#   FLOQUET_INSTANCES=80 sbatch slurm_floquet_40i_60k_1000s.sh q50-noise /path/cal.json
#
#   FLOQUET_INSTANCES : number of parallel gate instances        (default 40)
#   FLOQUET_KICKS     : circuits per instance (kick counts 0..K-1) (default 60)
#   FLOQUET_SHOTS     : shots per circuit                         (default 1000)
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

# --------------- tunables (env-overridable) ---------------
NUM_INSTANCES="${FLOQUET_INSTANCES:-40}"
NUM_KICKS="${FLOQUET_KICKS:-60}"
NUM_SHOTS="${FLOQUET_SHOTS:-1000}"

# --------------- args ---------------
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

# --------------- output dir ---------------
mkdir -p "${HPCQC_ROOT}/slurm_logs"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${HPCQC_ROOT}/results/floquet_${MODE//-/_}_${TS}_job${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "=== Floquet ${NUM_INSTANCES}-instance parallel run (${MODE}) ==="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "CPUs/task : ${SLURM_CPUS_PER_TASK}"
echo "Instances : ${NUM_INSTANCES}"
echo "Kicks     : ${NUM_KICKS}  (circuits per instance)"
echo "Shots     : ${NUM_SHOTS}"
echo "Container : ${HPCQC_CPU_CONTAINER}"
echo "Wrapper   : ${HPCQC_CPU_WRAPPER}"
echo "HPCQC root: ${HPCQC_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "Calib.    : ${CAL_PATH:-(noiseless, not used)}"
echo "Started   : $(date)"
echo

# --------------- env propagated into the container ---------------
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
export SINGULARITYENV_QISKIT_IN_PARALLEL=TRUE

# Cap external thread pools so each of the N Pool workers stays
# single-threaded. Without these, NumPy/BLAS (and Aer) each fan out to
# many threads per worker, and N workers * many-threads oversubscribes
# the 128-core node and slows everything down. Aer's own thread pool is
# additionally capped via max_parallel_threads=1 in floquet_runner.py.
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_NUMEXPR_NUM_THREADS=1

# --------------- calibration CLI arg ---------------
CAL_ARG=""
if [[ "$MODE" == "q50-noise" ]]; then
    CAL_ARG="--calibration ${CAL_PATH}"
fi

# --------------- single srun -> single container -> Pool inside ---------------
srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/floquet_runner.py" \
        --backend "${MODE}" \
        ${CAL_ARG} \
        --output-dir "${OUTDIR}" \
        --num-instances "${NUM_INSTANCES}" \
        --num-max-kicks "${NUM_KICKS}" \
        --num-shots "${NUM_SHOTS}"

EXIT_CODE=$?

echo
echo "=== Finished: $(date)  exit=${EXIT_CODE} ==="
echo
echo "Aggregate + plot:"
echo "  srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \\"
echo "      python3 ${HPCQC_ROOT}/aggregate_floquet.py ${OUTDIR}"

exit ${EXIT_CODE}
