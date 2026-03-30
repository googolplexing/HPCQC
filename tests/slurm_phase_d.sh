#!/bin/bash
#SBATCH --job-name=phase_d_val
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/phase_d_val.o%j
#SBATCH --error=slurm_logs/phase_d_val.e%j

# Phase D validation — runs inside the container on LUMI standard-g
# Usage: sbatch tests/slurm_phase_d.sh
#
# Expected runtime: < 2 minutes
# Expected output:  35/35 PASSED in slurm_logs/phase_d_val.o<jobid>
#
# This validates Steps 1-3 of Phase D (types, schema, quality gate).
# Export format tests (V15-V20) will be added in Step 4.

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== Phase D Validation — LUMI container run ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_GPU_CONTAINER}"
echo "Started:   $(date)"
echo ""

# Run Phase D validation inside container
srun --cpu-bind=${HPCQC_GPU_MASK} \
    ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/phase_d_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

# Also verify jsonschema is available (Phase D requirement from GREEN-REQ-001)
echo ""
echo "--- Container package check ---"
srun --cpu-bind=${HPCQC_GPU_MASK} \
    ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 -c "
import jsonschema, pyarrow, h5py
print(f'jsonschema: {jsonschema.__version__}')
print(f'pyarrow:    {pyarrow.__version__}')
print(f'h5py:       {h5py.__version__}')
print('All Phase D export dependencies present.')
"

exit ${EXIT_CODE}
