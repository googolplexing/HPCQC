#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=w1_manual_runnerqubits
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/w1_manual_runnerqubits.o%j
#SBATCH --error=slurm_logs/w1_manual_runnerqubits.e%j
#
# W1.6 STEP-1 confirmatory (RED-RESP-GATE2 §4 Step 1 / RED-CLARIFICATION-STEP1):
# the cross-implementation (idle-model) residual MEASUREMENT on a COMMON
# placement. Runs the CANONICAL examples/byo/floquet_dtc_q10_sweep.yaml but with
# physical_qubits PINNED (via run_sweep --physical-qubits, in-memory) to the
# runner reference's self-selected set QB1,2,5,6,7,9,10,11,12,13 — so both arms
# sit on identical qubits and the only difference left is the two idle
# implementations (runner ALAP self-scheduling vs the sweep's PadDelay +
# RelaxationNoisePass).
#
# NOT the 5-sigma gate: this is a MEASUREMENT judged against the PRE-COMMITTED
# ceiling (RED-APPROVAL-STEP1: floor 0.02, max rel-dev <= 2%). The verifier runs
# in --mode step1-residual; the residual it measures BECOMES sigma_sys for the
# later Option-1 gate. Converged (<= 2%) -> exit 0; over -> exit 1 = STOP (open
# the idle-implementation finding; do NOT widen the ceiling, RED-CLARIFICATION §5).
#
# Output-dir is JOB-STAMPED (sweep_output/w1_manual_runner_qubits_${SLURM_JOB_ID}):
# a unique per-run workspace, so the campaign_manifest is always fresh (no
# resume) — what we want for a one-off, and it also prevents the
# positional-manifest cross-config footgun (a stale manifest from a different
# placement skipping tasks by position). The --physical-qubits override does NOT
# touch grid expansion / task_ids / the manifest: expand_grid never reads
# physical_qubits; placements multiply execution UNITS inside _execute_byo_group,
# not the grid TASKS. (--mem=0 could be added to silence the RealMemory WARN and
# exercise the allocation-aware path; omitted here to match the gate's resource
# setup exactly so the comparison is clean. Cap is correct either way: 80 units,
# 1 wave.)
#
# Usage: sbatch tests/slurm_w1_manual_runner_qubits.sh
# Expected: verifier prints "VERDICT : CONVERGED" and exits 0 if the two idle
#   implementations agree within 2% on the common placement.

set -uo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"
mkdir -p slurm_logs

GATE_YAML="${HPCQC_ROOT}/examples/byo/floquet_dtc_q10_sweep.yaml"
REFERENCE="${HPCQC_ROOT}/examples/reference/floquet_dtc_q10_autocorr.csv"
VERIFIER="${HPCQC_ROOT}/tests/_w1_z_comb_verify.py"
RUNNER_QUBITS="QB1,QB2,QB5,QB6,QB7,QB9,QB10,QB11,QB12,QB13"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/sweep_output/w1_manual_runner_qubits_${SLURM_JOB_ID}"
REPORT="${OUTPUT_DIR}/w1_manual_runner_qubits_report.csv"
FLOOR="0.02"
MAX_REL_DEV="0.02"

echo "=== W1.6 Step-1 — manual runner-qubit residual (pinned placement) ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "YAML:      ${GATE_YAML} (unmodified; --physical-qubits + --output-dir overrides)"
echo "Placement: ${RUNNER_QUBITS} (runner self-selected set; solver bypassed)"
echo "Reference: ${REFERENCE}"
echo "OutputDir: ${OUTPUT_DIR}  (job-stamped; fresh manifest, no resume)"
echo "Ceiling:   max rel-dev <= ${MAX_REL_DEV}; floor ${FLOOR}  (PRE-COMMITTED, RED-APPROVAL)"
echo "Started:   $(date)"
echo ""

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"

cd "${SLURM_SUBMIT_DIR}"

# ── (1) Sweep: canonical YAML, placement PINNED to the runner set, clean
#        job-stamped workdir. ──
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 -m lumi_hpc_qc.sweep.run_sweep "${GATE_YAML}" \
        --output-dir "${OUTPUT_DIR}" \
        --physical-qubits "${RUNNER_QUBITS}" \
    || { echo "ERROR: sweep step failed" >&2; exit 2; }

echo ""
echo "=== Locate device-calibrated aggregated autocorrelator (runner qubits) ==="
# One placement -> exactly one device_calibrated aggregated dat. The per-placement
# subdir encodes the pinned qubits; glob and assert exactly one match.
shopt -s nullglob
CANDIDATES=( "${OUTPUT_DIR}"/byo_dat/floquet_dtc/*/device_calibrated/aggregated_autocorr.dat )
shopt -u nullglob
if [ "${#CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly 1 device_calibrated aggregated_autocorr.dat under" >&2
    echo "       ${OUTPUT_DIR}/byo_dat/floquet_dtc/*/device_calibrated/, found ${#CANDIDATES[@]}:" >&2
    printf '       %s\n' "${CANDIDATES[@]}" >&2
    exit 2
fi
CANDIDATE="${CANDIDATES[0]}"
echo "Candidate: ${CANDIDATE}"
echo ""

echo "=== (2) Step-1 residual MEASUREMENT vs the pre-committed ceiling ==="
# Measurement, NOT a 5-sigma gate. exit 0 = converged (<= ceiling) -> the
# reported residual is sigma_sys; exit 1 = over ceiling -> STOP; exit 3 =
# structural. Capture RC so the bank guidance is always emitted.
RC=0
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 "${VERIFIER}" \
        --mode step1-residual \
        --candidate "${CANDIDATE}" \
        --reference "${REFERENCE}" \
        --candidate-seeds 40 \
        --reference-seeds 40 \
        --floor "${FLOOR}" \
        --max-rel-dev "${MAX_REL_DEV}" \
        --report "${REPORT}" \
    || RC=$?

echo ""
echo "Bank (EITHER outcome — fix-provenance / sigma_sys evidence):"
echo "  scripts/bank_evidence.sh ${SLURM_JOB_ID} w1_manual_runner_qubits-${SLURM_JOB_ID} \\"
echo "      ${CANDIDATE} \\"
echo "      ${REPORT}"
echo "  (optionally cp the .dat to a self-identifying name, e.g."
echo "   w1_manual_runner_qubits_aggregated_autocorr.dat, then bank that copy.)"
if [ "${RC}" -eq 0 ]; then
    echo "VERDICT: CONVERGED (<= ${MAX_REL_DEV}) -> carry the measured sigma_sys to the Option-1 gate."
elif [ "${RC}" -eq 1 ]; then
    echo "VERDICT: NOT CONVERGED (> ${MAX_REL_DEV}) -> STOP: open the idle-implementation finding; do NOT widen the ceiling (RED-CLARIFICATION §5)." >&2
else
    echo "VERDICT: STRUCTURAL error (rc=${RC})." >&2
fi
echo "Finished: $(date)"
exit ${RC}
