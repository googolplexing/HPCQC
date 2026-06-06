#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# SMOKE: 2-node end-to-end shard->merge dress rehearsal for the 200-placement
# spatial survey, on a 4-chain / 2-seed slice. This is a PIPELINE check, not a
# science run. It exercises, on real device-calibrated dispatches:
#   * today's calibration loading + placement resolution (4 chains, 24 qubits),
#   * the per-qubit autocorrelator producer (P2) on the device-calibrated path,
#   * the cross-node fan-out: 2 shard ranks -> ONE merge, which is the FIRST
#     production use of the P3 ByoAutocorrReducer's per-qubit aggregation,
#   * 2 seeds (num_seeds >= 2) so the merge runs in the short-count-guard-
#     PROTECTED regime AND does real seed mean/sem (1 seed -> trivial sem=0).
# After it is green, render the partial per-site surface with map_dtc --per-qubit
# (24/52 qubits), then launch the full 32-node survey.
#
# This wraps the proven generic launcher (slurm_sweep_multinode.sh): same shard
# /merge body, but with the CORRECT account hardcoded (so it does not depend on
# env.sh being sourced in the submit shell) and the smoke slice as the default
# config (no stale-default footgun). Run:  sbatch smoke4_2node_spatial_survey_device_calibrated.sh
#
#SBATCH --job-name=smoke4_2node
#SBATCH --account=project_462001289
#SBATCH --partition=standard
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/smoke4_2node.o%j
#SBATCH --error=slurm_logs/smoke4_2node.e%j

set -uo pipefail

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/env.sh"

HPCQC_ROOT="${HPCQC_ROOT:-${SLURM_SUBMIT_DIR}}"
mkdir -p "${HPCQC_ROOT}/slurm_logs"
cd "${HPCQC_ROOT}" || { echo "cannot cd to ${HPCQC_ROOT}"; exit 2; }

# CPU standard partition, NO inter-node MPI: one independent engine per node,
# each writing its own shard to the shared FS. Override the GPU-oriented NIC
# policy env.sh sets, and run srun --mpi=none so no PMI/MPI bootstrap happens.
export MPICH_OFI_NIC_POLICY=NUMA
export MPICH_GPU_SUPPORT_ENABLED=0

export HPCQC_SWEEP_CONFIG="${HPCQC_SWEEP_CONFIG:-examples/byo/floquet_dtc_smoke4_q10_device_calibrated_100shots.yaml}"
export HPCQC_SWEEP_OUTDIR="${HPCQC_SWEEP_OUTDIR:-${HPCQC_ROOT}/results/smoke4_2node_job${SLURM_JOB_ID:-local}}"
mkdir -p "${HPCQC_SWEEP_OUTDIR}"

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export SINGULARITYENV_ROCR_VISIBLE_DEVICES=""
export SINGULARITYENV_HSA_TOOLS_LIB=""
export SINGULARITYENV_PYTHONUNBUFFERED=1
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1
export SINGULARITYENV_MPICH_OFI_NIC_POLICY=NUMA
export SINGULARITYENV_MPICH_GPU_SUPPORT_ENABLED=0

echo "================================================================"
echo " SMOKE: 2-node shard->merge dress rehearsal (4 chains, 2 seeds)"
echo "================================================================"
echo "Job ID     : ${SLURM_JOB_ID:-<none>}"
echo "Nodes      : ${SLURM_NNODES:-?} (one engine rank per node)"
echo "CPUs/node  : ${SLURM_CPUS_PER_TASK:-?}"
echo "Config     : ${HPCQC_SWEEP_CONFIG}"
echo "OutDir     : ${HPCQC_SWEEP_OUTDIR}"
echo "Container  : ${HPCQC_CPU_CONTAINER}"
echo "Started    : $(date)"
echo

echo "=== Step 1: shard runs (one rank per node) ==="
srun --ntasks-per-node=1 --mpi=none bash -c '
    export SINGULARITYENV_HPCQC_SWEEP_SHARD=1
    export SINGULARITYENV_SLURM_NODEID="${SLURM_NODEID}"
    export SINGULARITYENV_SLURM_NNODES="${SLURM_NNODES}"
    echo "  rank ${SLURM_NODEID}/${SLURM_NNODES} on $(hostname): shard run starting"
    "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
        python3 -u -m lumi_hpc_qc.sweep.run_sweep \
            "${HPCQC_SWEEP_CONFIG}" --output-dir "${HPCQC_SWEEP_OUTDIR}"
'
SHARD_RC=$?
if [ "${SHARD_RC}" -ne 0 ]; then
    echo "=== Shard runs FAILED (exit ${SHARD_RC}) - NOT merging ==="
    echo "Finished   : $(date)"
    exit "${SHARD_RC}"
fi

echo
echo "=== Step 2: merge rank shards (first production use of P3 per-qubit reducer) ==="
"${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 -u "${HPCQC_ROOT}/scripts/merge_sweep_shards.py" \
        --output-dir "${HPCQC_SWEEP_OUTDIR}" \
        --config "${HPCQC_SWEEP_CONFIG}" \
        --nranks "${SLURM_NNODES}"
MERGE_RC=$?

echo
if [ "${MERGE_RC}" -eq 0 ]; then
    echo "=== DONE (exit 0) - merged sweep in ${HPCQC_SWEEP_OUTDIR} ==="
    echo "    sweep.h5 + byo_dat/ (incl. aggregated_autocorr_perqubit.dat) + campaign_manifest.json"
else
    echo "=== MERGE FAILED (exit ${MERGE_RC}) - rank shards left in ${HPCQC_SWEEP_OUTDIR} ==="
fi
echo "Finished   : $(date)"
exit "${MERGE_RC}"
