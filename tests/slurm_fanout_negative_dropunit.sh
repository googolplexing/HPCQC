#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# NEGATIVE + positive gate for the PRODUCTION merge CLI on the battery path
# (RED-RULING-MERGE-CLI-FOLLOWUP §5 matrix). Runs tests/fanout_negative_dropunit.py
# inside the LUMI qiskit container: 2 battery shard runs, then the production CLI
# (scripts/merge_sweep_shards.py) on the COMPLETE shards (must exit 0, reducer=
# BatteryReducer) and on a SINGLE-UNIT-dropped copy (must fail loud, INCOMPLETE).
# Single node — ranks are subprocesses; the drop + merge are allocation-shape-
# independent.
#
# Light job (tiny fixture, a few subprocesses): submit to the SMALL partition.
# Usage:
#   sbatch --partition=small --cpus-per-task=32 tests/slurm_fanout_negative_dropunit.sh
#   # then:  tail -f slurm_logs/fanout_neg.o<JOBID>
#
#SBATCH --job-name=fanout_neg
#SBATCH --account=project_462000055
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/fanout_neg.o%j
#SBATCH --error=slurm_logs/fanout_neg.e%j

set -uo pipefail

export LUMI_QISKIT_SINGULARITY_CONTAINER_PATH="${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH:-/appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif}"
export WRAPPER_PATH="${WRAPPER_PATH:-/appl/local/quantum/qiskit/run-singularity}"

HPCQC_ROOT="${HPCQC_ROOT:-${SLURM_SUBMIT_DIR}}"
mkdir -p "${HPCQC_ROOT}/slurm_logs"
cd "${HPCQC_ROOT}" || { echo "cannot cd to ${HPCQC_ROOT}"; exit 2; }

echo "================================================================"
echo " Workstream-B production-CLI battery negative gate (single-unit drop)"
echo "================================================================"
echo "Job ID     : ${SLURM_JOB_ID:-<none>}"
echo "Node       : $(hostname)"
echo "HPCQC root : ${HPCQC_ROOT}"
echo "Container  : ${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH}"
echo "Started    : $(date)"
echo

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export SINGULARITYENV_ROCR_VISIBLE_DEVICES=""
export SINGULARITYENV_HSA_TOOLS_LIB=""
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1

NEG_WORKDIR="${HPCQC_ROOT}/results/fanout_neg_job${SLURM_JOB_ID:-local}"
mkdir -p "${NEG_WORKDIR}"
export SINGULARITYENV_NEG_WORKDIR="${NEG_WORKDIR}"

echo "═══ Running the battery production-CLI negative gate ═══"
echo
"${WRAPPER_PATH}" "${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH}" \
    python3 -u tests/fanout_negative_dropunit.py
RC=$?

echo
if [ "${RC}" -eq 0 ]; then
    echo "═══ NEG-GATE PASSED (exit 0) ═══"
else
    echo "═══ NEG-GATE FAILED (exit ${RC}) — artifacts in ${NEG_WORKDIR} ═══"
fi
echo "Finished   : $(date)"
exit "${RC}"
