#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# =============================================================================
# Mode B Controller — automated VQE orchestration with crash recovery
# =============================================================================
#
# Runs on a minimal CPU allocation (small partition, 1 task, 1 core).
# Submits the actual VQE as a child job to standard-g, monitors it, and
# automatically resubmits from the latest checkpoint on failure/timeout.
#
# No container, no GPU, no Python libs — pure bash + sbatch/squeue.
#
# Usage:
#   ./tests/launch_mode_b.sh configs/byo_tfim_8q.yaml
#   ./tests/launch_mode_b.sh configs/heisenberg_3x4_12q.yaml --walltime 02:00:00
# =============================================================================

#SBATCH --job-name=modeb_controller
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --output=slurm_logs/modeb_ctrl.o%j
#SBATCH --error=slurm_logs/modeb_ctrl.e%j

set -euo pipefail

# ── Source environment config ──
source "${SLURM_SUBMIT_DIR}/env.sh"

# ── Configuration (from env vars or defaults) ──
CONFIG_YAML="${MODEB_CONFIG:-configs/byo_tfim_8q.yaml}"
MAX_RETRIES="${MODEB_MAX_RETRIES:-3}"
POLL_INTERVAL="${MODEB_POLL_INTERVAL:-30}"
CHILD_WALLTIME="${MODEB_CHILD_WALLTIME:-01:00:00}"
CHECKPOINT_DIR="${HPCQC_ROOT}/checkpoints"

EXPERIMENT_ID="modeb_$(date +%s)_${SLURM_JOB_ID}"

# Directories
mkdir -p "${CHILD_SCRIPT_DIR}" "${HPCQC_ROOT}/slurm_logs" "${CHECKPOINT_DIR}"

# ── Banner ──
echo "========================================================================"
echo "  Mode B Controller"
echo "========================================================================"
echo "  Controller job: ${SLURM_JOB_ID}"
echo "  Node: $(hostname)"
echo "  Date: $(date)"
echo "  Config: ${CONFIG_YAML}"
echo "  Project: ${HPCQC_ROOT}"
echo "  GPU container: ${HPCQC_GPU_CONTAINER}"
echo "  GPU wrapper: ${HPCQC_GPU_WRAPPER}"
echo "  Max retries: ${MAX_RETRIES}"
echo "  Poll interval: ${POLL_INTERVAL}s"
echo "  Child walltime: ${CHILD_WALLTIME}"
echo "  Experiment ID: ${EXPERIMENT_ID}"
echo "========================================================================"
echo ""

# ── Generate child VQE script ──
generate_child_script() {
    local attempt=$1
    local runner=$2
    local extra_env="${3:-}"
    local script_path="${CHILD_SCRIPT_DIR}/child_attempt${attempt}.sh"

    cat > "${script_path}" << CHILD_EOF
#!/bin/bash
#SBATCH --job-name=modeb_child_a${attempt}
#SBATCH --partition=${HPCQC_GPU_PARTITION}
#SBATCH --time=${CHILD_WALLTIME}
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --output=${HPCQC_ROOT}/slurm_logs/modeb_child_a${attempt}.o%j
#SBATCH --error=${HPCQC_ROOT}/slurm_logs/modeb_child_a${attempt}.e%j

export SINGULARITYENV_PROJECT_DIR=${HPCQC_ROOT}
export SINGULARITYENV_CONFIG_PATH=${HPCQC_ROOT}/${CONFIG_YAML}
${extra_env}

SLURM_START_EPOCH=\$(date +%s)
echo "=== Mode B child attempt ${attempt} ==="
echo "Job \${SLURM_JOB_ID} on \${SLURM_NODELIST}"
echo "Date: \$(date)"
export SINGULARITYENV_SLURM_START_EPOCH=\$SLURM_START_EPOCH

srun --cpu-bind=${HPCQC_GPU_MASK} \\
  ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \\
  python ${HPCQC_ROOT}/tests/${runner}

echo "Completed: \$(date), wall: \$(( \$(date +%s) - SLURM_START_EPOCH ))s"
CHILD_EOF

    chmod +x "${script_path}"
    echo "${script_path}"
}

# ── Find latest checkpoint ──
find_checkpoint() {
    local latest=$(ls -t "${CHECKPOINT_DIR}"/*_iter*.json 2>/dev/null | head -1)
    echo "${latest:-}"
}

# ── Monitor child job ──
monitor_child() {
    local child_id=$1
    local start_time=$(date +%s)

    while true; do
        local state=$(squeue -j "${child_id}" --format="%T" --noheader 2>/dev/null | head -1 | tr -d ' ')

        if [ -z "$state" ]; then
            state=$(sacct -j "${child_id}" --format=State --noheader --parsable2 2>/dev/null | head -1 | tr -d ' ')
        fi

        local elapsed=$(( $(date +%s) - start_time ))
        local elapsed_min=$(( elapsed / 60 ))

        # Read progress file if available
        local progress=""
        local progress_file=$(ls -t "${HPCQC_ROOT}"/results/*_progress.json 2>/dev/null | head -1)
        if [ -n "${progress_file:-}" ] && [ -f "${progress_file}" ]; then
            local iter=$(python3 -c "import json; d=json.load(open('${progress_file}')); print(d.get('iterations_completed','?'))" 2>/dev/null || echo "?")
            local best_e=$(python3 -c "import json; d=json.load(open('${progress_file}')); print(f\"{d.get('best_energy','?'):.6f}\")" 2>/dev/null || echo "?")
            progress="  iter=${iter} best_E=${best_e}"
        fi

        echo "  [$(date +%H:%M:%S)] Child ${child_id}: ${state:-UNKNOWN} (${elapsed_min}m)${progress}"

        case "${state}" in
            COMPLETED)  return 0 ;;
            FAILED)     return 1 ;;
            TIMEOUT)    return 2 ;;
            CANCELLED*) return 3 ;;
        esac

        sleep "${POLL_INTERVAL}"
    done
}

# ── Main orchestration loop ──
CHILD_IDS=()
ATTEMPT=0
SUCCESS=false

while [ ${ATTEMPT} -le ${MAX_RETRIES} ]; do
    ATTEMPT=$((ATTEMPT + 1))
    echo ""
    echo "── Attempt ${ATTEMPT}/$((MAX_RETRIES + 1)) ──"

    CHECKPOINT=$(find_checkpoint)

    if [ ${ATTEMPT} -gt 1 ] && [ -n "${CHECKPOINT}" ]; then
        echo "  Resuming from checkpoint: ${CHECKPOINT}"
        EXTRA_ENV="export SINGULARITYENV_RESUME_CHECKPOINT=${CHECKPOINT}"
        RUNNER="resume_runner.py"
    else
        [ ${ATTEMPT} -gt 1 ] && echo "  No checkpoint found — starting fresh"
        EXTRA_ENV=""
        RUNNER="vqe_runner.py"
    fi

    SCRIPT=$(generate_child_script ${ATTEMPT} "${RUNNER}" "${EXTRA_ENV}")
    echo "  Script: ${SCRIPT}"

    CHILD_ID=$(sbatch "${SCRIPT}" 2>&1 | grep -oP '\d+$')
    if [ -z "${CHILD_ID}" ]; then
        echo "  ERROR: sbatch failed"
        sleep 60
        continue
    fi

    CHILD_IDS+=("${CHILD_ID}")
    echo "  Submitted child: ${CHILD_ID}"
    echo ""

    monitor_child "${CHILD_ID}"
    EXIT_CODE=$?

    case ${EXIT_CODE} in
        0)
            echo ""
            echo "  Child ${CHILD_ID} COMPLETED"
            SUCCESS=true
            break
            ;;
        1)
            echo ""
            echo "  Child ${CHILD_ID} FAILED"
            sacct -j "${CHILD_ID}" --format=JobID,State,ExitCode,Elapsed --noheader 2>/dev/null | head -3
            ;;
        2)
            echo ""
            echo "  Child ${CHILD_ID} TIMEOUT"
            ;;
        3)
            echo ""
            echo "  Child ${CHILD_ID} CANCELLED — aborting"
            break
            ;;
    esac

    if [ ${ATTEMPT} -le ${MAX_RETRIES} ]; then
        echo "  Retrying in 10 seconds..."
        sleep 10
    fi
done

# ── Summary ──
echo ""
echo "========================================================================"
echo "  Mode B Summary"
echo "========================================================================"
echo "  Controller: ${SLURM_JOB_ID}"
echo "  Attempts: ${ATTEMPT}"
echo "  Child IDs: ${CHILD_IDS[*]}"
echo "  Success: ${SUCCESS}"

LAST_CHILD="${CHILD_IDS[-1]:-}"
if [ -n "${LAST_CHILD}" ]; then
    echo ""
    echo "  Last child output:"
    CHILD_OUT=$(ls -t "${HPCQC_ROOT}"/slurm_logs/modeb_child_*.o* 2>/dev/null | head -1)
    if [ -n "${CHILD_OUT:-}" ]; then
        grep -E "(Best energy|Exact energy|relative error|PASSED|FAILED|RESULT|Iterations)" "${CHILD_OUT}" 2>/dev/null | head -10
    fi
fi

echo "========================================================================"
echo "Controller finished: $(date)"
