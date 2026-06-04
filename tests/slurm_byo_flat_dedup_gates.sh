#!/bin/bash
# BYO flat-dispatch acceptance gates (RED-RULING-BYO-FLAT-DISPATCH-AND-NOISELESS-
# DEDUP). Runs, in the LUMI qiskit container, on the multi-observable fixture:
#
#   GATE 2 (flat byte-identity): fanout_byte_identity_gate.py single vs 2- vs
#           3-rank. The 3-rank (non-dividing) merge reconstructing all units IS
#           the in-engine proof the (env,observable) strata distributes the
#           un-folded families correctly across ranks.
#   GATE 4 (§5.4 dedup):         byo_dedup_byte_identity_gate.py dedup off vs on,
#           byte-identical output + dedup proven to engage.
#
# GATE 1 (single-observable byte-identity, A+B == pre-B) is the existing W1
# canary (tests/slurm_w1_canary.sh) — run it separately; it compares to the
# pre-B oracle.
#
# Account comes from SBATCH_ACCOUNT (env.sh) — `source env.sh` before sbatch.
# Usage:
#   source env.sh && sbatch tests/slurm_byo_flat_dedup_gates.sh
#   cat $(ls -t slurm_logs/byo_gates.o* | head -1)
#
#SBATCH --job-name=byo_gates
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=224G
#SBATCH --time=00:30:00
#SBATCH --output=slurm_logs/byo_gates.o%j
#SBATCH --error=slurm_logs/byo_gates.e%j

set -uo pipefail

HPCQC_ROOT="${HPCQC_ROOT:-${SLURM_SUBMIT_DIR}}"
cd "${HPCQC_ROOT}" || { echo "cannot cd to ${HPCQC_ROOT}"; exit 2; }
source env.sh
mkdir -p slurm_logs

CFG="examples/byo/floquet_dtc_q10_multiobs_gate.yaml"
WORK="results/byo_gates_${SLURM_JOB_ID:-local}"
mkdir -p "${WORK}/fanout" "${WORK}/dedup"

export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_PYTHONUNBUFFERED=1

echo "================================================================"
echo " BYO flat-dispatch gates (multi-observable)"
echo " Job ${SLURM_JOB_ID:-<none>} on $(hostname) — $(date)"
echo " Config: ${CFG}"
echo "================================================================"

echo
echo "═══ GATE 2: flat byte-identity (single vs 2-rank vs 3-rank) ═══"
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 -u tests/fanout_byte_identity_gate.py \
        --path byo --config "${CFG}" --workdir "${WORK}/fanout"
RC2=$?
echo "GATE2_RC=${RC2}"

echo
echo "═══ GATE 4: §5.4 noiseless-dedup byte-identity (dedup off vs on) ═══"
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 -u tests/byo_dedup_byte_identity_gate.py \
        --config "${CFG}" --workdir "${WORK}/dedup"
RC4=$?
echo "GATE4_RC=${RC4}"

echo
echo "═══ SUMMARY: gate2(flat)=${RC2}  gate4(dedup)=${RC4} ═══"
if [ "${RC2}" -eq 0 ] && [ "${RC4}" -eq 0 ]; then
    echo "ALL GATES PASSED"
else
    echo "A GATE FAILED — artifacts under ${WORK}"
fi
echo "Finished $(date)"
exit $(( RC2 + RC4 ))
