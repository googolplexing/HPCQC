#!/bin/bash
#SBATCH --job-name=phase_d_export
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/phase_d_export.o%j
#SBATCH --error=slurm_logs/phase_d_export.e%j

# Phase D export validation — tests all 6 export formats on the container.
# HDF5 (h5py 3.16.0) and Parquet (pyarrow 23.0.1) require the container.
# CSV, JSONL, and NPZ also run locally without the container.
#
# Usage:    sbatch tests/slurm_phase_d_export.sh
# Expected: 35/35 PASSED
# Runtime:  < 2 minutes

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== Phase D Export Validation — LUMI container run ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_GPU_CONTAINER}"
echo "Started:   $(date)"
echo ""

srun ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/phase_d_export_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
