#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Run N Floquet gate instances in parallel on ONE LUMI standard node.
# Single srun, single container, multiprocessing.Pool inside Python.
#
# Defaults: 40 instances, 60 kicks/circuits per instance, 1000 shots.
#
# Usage:
#   sbatch slurm_floquet_40i_60k_1000s.sh noiseless
#   sbatch slurm_floquet_40i_60k_1000s.sh logical-gates     /path/cal.json
#   sbatch slurm_floquet_40i_60k_1000s.sh device-calibrated /path/cal.json
#   sbatch slurm_floquet_40i_60k_1000s.sh iqm-fake-backend
#
# Env-overridable tunables (defaults shown):
#   FLOQUET_INSTANCES=40   FLOQUET_KICKS=60   FLOQUET_SHOTS=1000
#   FLOQUET_T2_MODE=ramsey            (device-calibrated only: ramsey|echo)
#   FLOQUET_IQM_DEVICE=aphrodite      (iqm-fake-backend only: aphrodite|apollo)
#   FLOQUET_PRX_NS / FLOQUET_CZ_NS / FLOQUET_MEASURE_NS  (override JSON durations)
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

NUM_INSTANCES="${FLOQUET_INSTANCES:-40}"
NUM_KICKS="${FLOQUET_KICKS:-60}"
NUM_SHOTS="${FLOQUET_SHOTS:-1000}"
T2_MODE="${FLOQUET_T2_MODE:-ramsey}"
IQM_DEVICE="${FLOQUET_IQM_DEVICE:-aphrodite}"

NOISE_SOURCE="${1:-}"
CAL_PATH="${2:-}"

case "$NOISE_SOURCE" in
  noiseless|iqm-fake-backend) NEEDS_CAL=0 ;;
  logical-gates|device-calibrated) NEEDS_CAL=1 ;;
  *)
    echo "ERROR: first arg must be one of: noiseless | logical-gates | device-calibrated | iqm-fake-backend"
    echo "got '${NOISE_SOURCE}'"
    exit 2 ;;
esac

if [[ "$NEEDS_CAL" == "1" ]]; then
    if [[ -z "$CAL_PATH" ]]; then
        echo "ERROR: ${NOISE_SOURCE} needs a calibration JSON as 2nd arg"; exit 2
    fi
    if [[ ! -f "$CAL_PATH" ]]; then
        echo "ERROR: calibration file not found: $CAL_PATH"; exit 2
    fi
fi

mkdir -p "${HPCQC_ROOT}/slurm_logs"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${HPCQC_ROOT}/results/floquet_${NOISE_SOURCE//-/_}_${TS}_job${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "=== Floquet ${NUM_INSTANCES}-instance run (${NOISE_SOURCE}) ==="
echo "Job ID    : ${SLURM_JOB_ID}"
echo "Node      : $(hostname)"
echo "CPUs/task : ${SLURM_CPUS_PER_TASK}"
echo "Instances : ${NUM_INSTANCES}   Kicks: ${NUM_KICKS}   Shots: ${NUM_SHOTS}"
echo "Container : ${HPCQC_CPU_CONTAINER}"
echo "HPCQC root: ${HPCQC_ROOT}"
echo "Output dir: ${OUTDIR}"
echo "Calib.    : ${CAL_PATH:-(not used)}"
[[ "$NOISE_SOURCE" == "device-calibrated" ]] && echo "T2 mode   : ${T2_MODE}"
[[ "$NOISE_SOURCE" == "iqm-fake-backend" ]] && echo "IQM device: ${IQM_DEVICE}"
echo "Started   : $(date)"
echo

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export SINGULARITYENV_ROCR_VISIBLE_DEVICES=""
export SINGULARITYENV_HSA_TOOLS_LIB=""
export SINGULARITYENV_QISKIT_IN_PARALLEL=TRUE
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_NUMEXPR_NUM_THREADS=1

# Assemble optional CLI args.
EXTRA_ARGS=""
if [[ "$NEEDS_CAL" == "1" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --calibration ${CAL_PATH}"
fi
if [[ "$NOISE_SOURCE" == "device-calibrated" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --t2-mode ${T2_MODE}"
    [[ -n "${FLOQUET_PRX_NS:-}" ]]     && EXTRA_ARGS="${EXTRA_ARGS} --prx-time-ns ${FLOQUET_PRX_NS}"
    [[ -n "${FLOQUET_CZ_NS:-}" ]]      && EXTRA_ARGS="${EXTRA_ARGS} --cz-time-ns ${FLOQUET_CZ_NS}"
    [[ -n "${FLOQUET_MEASURE_NS:-}" ]] && EXTRA_ARGS="${EXTRA_ARGS} --measure-time-ns ${FLOQUET_MEASURE_NS}"
fi
if [[ "$NOISE_SOURCE" == "iqm-fake-backend" ]]; then
    EXTRA_ARGS="${EXTRA_ARGS} --iqm-device ${IQM_DEVICE}"
fi

srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/floquet_runner.py" \
        --noise-source "${NOISE_SOURCE}" \
        ${EXTRA_ARGS} \
        --output-dir "${OUTDIR}" \
        --num-instances "${NUM_INSTANCES}" \
        --num-max-kicks "${NUM_KICKS}" \
        --num-shots "${NUM_SHOTS}"

EXIT_CODE=$?
echo
echo "=== Finished: $(date)  exit=${EXIT_CODE} ==="
echo "Aggregate + plot:"
echo "  srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \\"
echo "      python3 ${HPCQC_ROOT}/aggregate_floquet.py ${OUTDIR}"
exit ${EXIT_CODE}
