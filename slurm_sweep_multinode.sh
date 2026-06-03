#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Multi-node sweep launcher for the Workstream-B cross-node fan-out
# (RED-RULING-WORKSTREAM-B-CROSS-NODE-FANOUT §3: a NEW, generically-named
# launcher — NOT a mode flag on the single-node launcher, which stays
# byte-untouched and is what the byte-identity gate compares against).
#
# It runs ONE engine process per node under HPCQC_SWEEP_SHARD=1: each node takes
# its stratified slice of the flat work-unit list (SLURM_NODEID / SLURM_NNODES)
# and writes its own sweep_rank{r}.h5 + campaign_manifest_rank{r}.json into the
# shared OUTDIR, deferring .dat aggregation. After all ranks finish, a single
# merge step unions the rank shards, asserts each (placement,env) seed series is
# complete, aggregates via the certified aggregate_byo_autocorr, and concats the
# manifests — producing sweep.h5 + byo_dat + campaign_manifest.json that are
# byte-identical to a single-node run of the same units (proven by
# tests/slurm_fanout_byte_identity_gate.sh).
#
# This launcher is sweep-type-agnostic at the shard/merge layer; the BYO reducer
# is what runs at merge. (The hamiltonian-battery reducer + its equivalence gate
# — Workstream-B item 3 — is what proves the layer is generic, not BYO-only.)
#
# Usage:
#   # size the allocation with --nodes; point at a sweep YAML:
#   sbatch --nodes=4 \
#       --export=ALL,HPCQC_SWEEP_CONFIG=examples/byo/floquet_dtc_q10_sweep.yaml \
#       slurm_sweep_multinode.sh
#   # change --nodes / --cpus-per-task to redistribute — no code change.
#   # then:  tail -f slurm_logs/sweep_multinode.o<JOBID>
#
#SBATCH --job-name=sweep_multinode
#SBATCH --account=project_462000055
#SBATCH --partition=standard
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --output=slurm_logs/sweep_multinode.o%j
#SBATCH --error=slurm_logs/sweep_multinode.e%j

set -uo pipefail

export LUMI_QISKIT_SINGULARITY_CONTAINER_PATH="${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH:-/appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif}"
export WRAPPER_PATH="${WRAPPER_PATH:-/appl/local/quantum/qiskit/run-singularity}"

HPCQC_ROOT="${HPCQC_ROOT:-${SLURM_SUBMIT_DIR}}"
mkdir -p "${HPCQC_ROOT}/slurm_logs"
cd "${HPCQC_ROOT}" || { echo "cannot cd to ${HPCQC_ROOT}"; exit 2; }

# Sweep config (a real default so it runs out-of-box) + a shared OUTDIR all ranks
# and the merge agree on (the engine ranks write sweep_rank{r}.h5 here; the merge
# reads them back). Both come through --output-dir so the YAML is never edited.
export HPCQC_SWEEP_CONFIG="${HPCQC_SWEEP_CONFIG:-examples/byo/floquet_dtc_q10_sweep.yaml}"
export HPCQC_SWEEP_OUTDIR="${HPCQC_SWEEP_OUTDIR:-${HPCQC_ROOT}/results/sweep_multinode_job${SLURM_JOB_ID:-local}}"
mkdir -p "${HPCQC_SWEEP_OUTDIR}"

# ── env propagated into the container (same for every rank). The PER-RANK
#    SLURM_NODEID/SLURM_NNODES are set INSIDE the srun command below, because at
#    batch-script scope $SLURM_NODEID is the batch step's value, not the task's. ──
export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export SINGULARITYENV_ROCR_VISIBLE_DEVICES=""
export SINGULARITYENV_HSA_TOOLS_LIB=""
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1

echo "================================================================"
echo " Workstream-B multi-node sweep (cross-node fan-out)"
echo "================================================================"
echo "Job ID     : ${SLURM_JOB_ID:-<none>}"
echo "Nodes      : ${SLURM_NNODES:-?} (one engine rank per node)"
echo "CPUs/node  : ${SLURM_CPUS_PER_TASK:-?}"
echo "Config     : ${HPCQC_SWEEP_CONFIG}"
echo "OutDir     : ${HPCQC_SWEEP_OUTDIR}"
echo "Container  : ${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH}"
echo "Started    : $(date)"
echo

# ── Step 1: shard run — one engine per node. Each task exports its OWN
#    SLURM_NODEID/SLURM_NNODES into the container ($SLURM_NODEID is per-task here)
#    plus HPCQC_SWEEP_SHARD=1, so the engine resolves (rank, nranks) = (NODEID,
#    NNODES), shards its slice, and writes sweep_rank{NODEID}.h5 + its manifest. ──
echo "═══ Step 1: shard runs (one rank per node) ═══"
srun --ntasks-per-node=1 bash -c '
    export SINGULARITYENV_HPCQC_SWEEP_SHARD=1
    export SINGULARITYENV_SLURM_NODEID="${SLURM_NODEID}"
    export SINGULARITYENV_SLURM_NNODES="${SLURM_NNODES}"
    echo "  rank ${SLURM_NODEID}/${SLURM_NNODES} on $(hostname): shard run starting"
    "${WRAPPER_PATH}" "${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH}" \
        python3 -u -m lumi_hpc_qc.sweep.run_sweep \
            "${HPCQC_SWEEP_CONFIG}" --output-dir "${HPCQC_SWEEP_OUTDIR}"
'
SHARD_RC=$?
if [ "${SHARD_RC}" -ne 0 ]; then
    echo "═══ Shard runs FAILED (exit ${SHARD_RC}) — NOT merging ═══"
    echo "Finished   : $(date)"
    exit "${SHARD_RC}"
fi

# ── Step 2: merge once (batch node, in-container for h5py). Unions the rank
#    shards, asserts completeness vs the config seed_list (fail-loud short-count
#    guard), aggregates, concats manifests -> sweep.h5 + byo_dat + manifest. ──
echo
echo "═══ Step 2: merge rank shards ═══"
"${WRAPPER_PATH}" "${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH}" \
    python3 -u "${HPCQC_ROOT}/scripts/merge_sweep_shards.py" \
        --output-dir "${HPCQC_SWEEP_OUTDIR}" \
        --config "${HPCQC_SWEEP_CONFIG}"
MERGE_RC=$?

echo
if [ "${MERGE_RC}" -eq 0 ]; then
    echo "═══ DONE (exit 0) — merged sweep in ${HPCQC_SWEEP_OUTDIR} ═══"
    echo "    sweep.h5 + byo_dat/ + campaign_manifest.json"
else
    echo "═══ MERGE FAILED (exit ${MERGE_RC}) — rank shards left in ${HPCQC_SWEEP_OUTDIR} ═══"
fi
echo "Finished   : $(date)"
exit "${MERGE_RC}"
