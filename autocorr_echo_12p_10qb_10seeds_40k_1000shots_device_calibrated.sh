#!/bin/bash
#SBATCH --job-name=autocorr_echo_12p_10qb_10s_40k_100s_sweep_multinode
#SBATCH --account=project_462001289
#SBATCH --partition=standard
#SBATCH --nodes=2
#SBATCH --mem=224G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:20:00
#SBATCH --output=slurm_logs/autocorr_echo_12p_10qb_10s_40k_100s_sweep_multinode.o%j
#SBATCH --error=slurm_logs/autocorr_echo_12p_10qb_10s_40k_100s_sweep_multinode.e%j

source "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}/env.sh"
export MPICH_OFI_NIC_POLICY=NUMA
export MPICH_GPU_SUPPORT_ENABLED=0
export HPCQC_SWEEP_CONFIG="${HPCQC_SWEEP_CONFIG:-examples/byo/floquet_dtc_q10_1000shots_echo_sweep_device_calibrated.yaml}"
export HPCQC_SWEEP_OUTDIR="${HPCQC_SWEEP_OUTDIR:-${HPCQC_ROOT}/results/autocorr_echo_12p_10qb_10s_40k_1000s_${SLURM_JOB_ID:-local}}"

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

echo "Job ID     : ${SLURM_JOB_ID:-<none>}"
echo "Nodes      : ${SLURM_NNODES:-?} (one engine rank per node)"
echo "CPUs/node  : ${SLURM_CPUS_PER_TASK:-?}"
echo "Config     : ${HPCQC_SWEEP_CONFIG}"
echo "OutDir     : ${HPCQC_SWEEP_OUTDIR}"
echo "Container  : ${HPCQC_CPU_CONTAINER}"
echo "Started    : $(date)"

# ── Step 1: shard run — one engine per node. Each task exports its OWN
#    SLURM_NODEID/SLURM_NNODES into the container ($SLURM_NODEID is per-task here)
#    plus HPCQC_SWEEP_SHARD=1, so the engine resolves (rank, nranks) = (NODEID,
#    NNODES), shards its slice, and writes sweep_rank{NODEID}.h5 + its manifest. ──
echo "═══ Step 1: shard runs (one rank per node) ═══"
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
    echo "═══ Shard runs FAILED (exit ${SHARD_RC}) — NOT merging ═══"
    echo "Finished   : $(date)"
    exit "${SHARD_RC}"
fi

# ── Step 2: merge once (batch node, in-container for h5py). Unions the rank
#    shards, selects the reducer by experiment type, asserts each EXTRACTED
#    group has its complete seed series (PARTIAL short-count guard — see header)
#    AND the unioned group SET == campaign_expected.json (option-(i) wholly-
#    absent-group guard; --nranks catches a whole-rank-file drop at discovery),
#    aggregates (BYO) / identity-reduces (battery), concats manifests. ──

echo
echo "═══ Step 2: merge rank shards ═══"
"${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 -u "${HPCQC_ROOT}/scripts/merge_sweep_shards.py" \
        --output-dir "${HPCQC_SWEEP_OUTDIR}" \
        --config "${HPCQC_SWEEP_CONFIG}" \
        --nranks "${SLURM_NNODES}"
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
