#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Byte-identity GATE for the Workstream-B cross-node fan-out (BYO path).
# Runs tests/fanout_byte_identity_gate.py inside the LUMI qiskit container: the
# 8-unit fixture single-node vs 2-rank vs 3-rank, asserting the merged .dat tree
# is byte-identical and the merged sweep.h5 /byo subtree matches at the dataset
# level. ONE node suffices — each rank is a separate engine process with
# HPCQC_SWEEP_SHARD=1 + SLURM_NNODES/SLURM_NODEID set by the gate, and a unit's
# output is a pure function of its seed (allocation-shape-independent).
#
# Usage:
#   sbatch tests/slurm_fanout_byte_identity_gate.sh
#   # then:  tail -f slurm_logs/fanout_gate.o<JOBID>
#
#SBATCH --job-name=fanout_gate
#SBATCH --account=project_462000055
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/fanout_gate.o%j
#SBATCH --error=slurm_logs/fanout_gate.e%j

set -uo pipefail

export LUMI_QISKIT_SINGULARITY_CONTAINER_PATH="${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH:-/appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif}"
export WRAPPER_PATH="${WRAPPER_PATH:-/appl/local/quantum/qiskit/run-singularity}"

HPCQC_ROOT="${HPCQC_ROOT:-${SLURM_SUBMIT_DIR}}"
mkdir -p "${HPCQC_ROOT}/slurm_logs"
cd "${HPCQC_ROOT}" || { echo "cannot cd to ${HPCQC_ROOT}"; exit 2; }

echo "================================================================"
echo " Workstream-B fan-out byte-identity gate"
echo "================================================================"
echo "Job ID     : ${SLURM_JOB_ID:-<none>}"
echo "Node       : $(hostname)"
echo "HPCQC root : ${HPCQC_ROOT}"
echo "Container  : ${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH}"
echo "Started    : $(date)"
echo

# ── env to propagate into the container ──
export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export SINGULARITYENV_ROCR_VISIBLE_DEVICES=""
export SINGULARITYENV_HSA_TOOLS_LIB=""
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1

# Gate workdir under the job's scratch (results/), so artifacts persist for
# inspection if the gate fails.
GATE_WORKDIR="${HPCQC_ROOT}/results/fanout_gate_job${SLURM_JOB_ID:-local}"
mkdir -p "${GATE_WORKDIR}"
export SINGULARITYENV_GATE_WORKDIR="${GATE_WORKDIR}"

echo "═══ Running the gate (single vs 2-rank vs 3-rank) ═══"
echo
"${WRAPPER_PATH}" "${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH}" \
    python3 -u tests/fanout_byte_identity_gate.py \
        --config examples/byo/floquet_dtc_q10_fanout_gate_8unit.yaml \
        --workdir "${GATE_WORKDIR}"
RC=$?

echo
if [ "${RC}" -eq 0 ]; then
    echo "═══ GATE PASSED (exit 0) ═══"
else
    echo "═══ GATE FAILED (exit ${RC}) — artifacts in ${GATE_WORKDIR} ═══"
fi
echo "Finished   : $(date)"
exit "${RC}"
