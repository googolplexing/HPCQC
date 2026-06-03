#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Acceptance gate for option (i) — the WHOLLY-ABSENT-group guard — on the
# PRODUCTION merge CLI (RED-RULING-PATCH43-VERIFY-AND-INVENTORY-DESIGN §3 Q4,
# §4.3). Runs tests/fanout_negative_dropgroup.py inside the LUMI qiskit
# container: 2 battery shard runs (which make the engine write
# campaign_expected.json), then the production CLI (scripts/merge_sweep_shards.py)
# three ways, all with --nranks present:
#   POSITIVE — complete shards + inventory -> exit 0, sweep.h5, group-set checked
#              silently (the engine's expected keys match extract's, no drift);
#   (i-a)    — drop a whole RANK FILE -> fail loud at discovery (file count !=
#              nranks);
#   (i-b)    — drop a whole (placement,env) GROUP from present files -> fail loud
#              at the group-set assert (THE blind-spot closure).
# Single node — ranks are subprocesses; the drops + merges are allocation-shape-
# independent.
#
# Light job (tiny fixture, a few subprocesses): submit to the SMALL partition.
# Usage:
#   sbatch --partition=small --cpus-per-task=32 tests/slurm_fanout_negative_dropgroup.sh
#   # then:  tail -f slurm_logs/fanout_dropgroup.o<JOBID>
#
#SBATCH --job-name=fanout_dropgroup
#SBATCH --account=project_462000055
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/fanout_dropgroup.o%j
#SBATCH --error=slurm_logs/fanout_dropgroup.e%j

set -uo pipefail

export LUMI_QISKIT_SINGULARITY_CONTAINER_PATH="${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH:-/appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif}"
export WRAPPER_PATH="${WRAPPER_PATH:-/appl/local/quantum/qiskit/run-singularity}"

HPCQC_ROOT="${HPCQC_ROOT:-${SLURM_SUBMIT_DIR}}"
mkdir -p "${HPCQC_ROOT}/slurm_logs"
cd "${HPCQC_ROOT}" || { echo "cannot cd to ${HPCQC_ROOT}"; exit 2; }

echo "================================================================"
echo " Option-(i) wholly-absent-group gate (i-a file drop + i-b group drop)"
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

NEG_WORKDIR="${HPCQC_ROOT}/results/fanout_dropgroup_job${SLURM_JOB_ID:-local}"
mkdir -p "${NEG_WORKDIR}"
export SINGULARITYENV_NEG_WORKDIR="${NEG_WORKDIR}"

echo "═══ Running the option-(i) wholly-absent-group gate ═══"
echo
"${WRAPPER_PATH}" "${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH}" \
    python3 -u tests/fanout_negative_dropgroup.py
RC=$?

echo
if [ "${RC}" -eq 0 ]; then
    echo "═══ DROPGROUP-GATE PASSED (exit 0) ═══"
else
    echo "═══ DROPGROUP-GATE FAILED (exit ${RC}) — artifacts in ${NEG_WORKDIR} ═══"
fi
echo "Finished   : $(date)"
exit "${RC}"
