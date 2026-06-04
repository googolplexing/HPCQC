#!/bin/bash
# RED-DIRECTIVE-PROBE-SKIP-WHEN-NON-BINDING §6 — LUMI A/B byte-identity gate.
# Runs the multi-observable fixture single-node TWICE on identical config —
# probe-skip active (default) vs HPCQC_FORCE_PROBE=1 (forced probe) — and asserts
# byte-identical .dat + /byo output AND that the two arms took distinct cap paths
# (skip:mem_non_binding vs probe:device_calibrated_VmHWM). Confirms the probe
# skip is physics-invariant. Allocation-aware --mem (no D3 RealMemory WARN).
#
# Account comes from SBATCH_ACCOUNT (env.sh) — `source env.sh` before sbatch.
# Usage:
#   source env.sh && sbatch tests/slurm_byo_probe_skip_gate.sh
#   cat $(ls -t slurm_logs/byo_probe_skip.o* | head -1)
#
#SBATCH --job-name=byo_probe_skip
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/byo_probe_skip.o%j
#SBATCH --error=slurm_logs/byo_probe_skip.e%j

set -uo pipefail

HPCQC_ROOT="${HPCQC_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "${HPCQC_ROOT}" || { echo "cannot cd to ${HPCQC_ROOT}"; exit 2; }
source env.sh
mkdir -p slurm_logs

CFG="examples/byo/floquet_dtc_q10_multiobs_gate.yaml"
WORK="results/byo_probe_skip_${SLURM_JOB_ID:-local}"
mkdir -p "${WORK}"

export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

echo "================================================================"
echo " BYO probe-skip A/B byte-identity gate (RED-DIRECTIVE-PROBE-SKIP)"
echo " Job ${SLURM_JOB_ID:-<none>} on $(hostname) — $(date)"
echo " Config: ${CFG}"
echo "================================================================"

srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 -u tests/byo_probe_skip_byte_identity_gate.py \
        --config "${CFG}" --workdir "${WORK}"
RC=$?

echo
echo "═══ SUMMARY: probe_skip_ab=${RC} ═══"
if [ "${RC}" -eq 0 ]; then
    echo "ALL GATES PASSED"
else
    echo "A GATE FAILED — artifacts under ${WORK}"
fi
echo "Finished $(date)"
exit ${RC}
