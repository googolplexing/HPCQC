#!/bin/bash
#SBATCH --job-name=qpy_export
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/qpy_export.o%j
#SBATCH --error=slurm_logs/qpy_export.e%j

# QPY export and V20 validation
# Rebuilds the optimised ansatz circuit from a completed VQE result JSON,
# serialises it to QPY format, and verifies the round-trip.
#
# Usage:
#   sbatch tests/slurm_qpy_export.sh <result_json>
#
# Example:
#   sbatch tests/slurm_qpy_export.sh \
#     results/byo/302cc5e01c5d_17118012_result.json
#
# Expected output:
#   V20 PASS: QPY round-trip verified — num_qubits=4, depth=..., num_parameters=0

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

RESULT_JSON="${1:-results/byo/302cc5e01c5d_17118012_result.json}"

echo "=== QPY Export — V20 Validation ==="
echo "Job ID:      ${SLURM_JOB_ID}"
echo "Node:        $(hostname)"
echo "Container:   ${HPCQC_GPU_CONTAINER}"
echo "Result JSON: ${RESULT_JSON}"
echo "Started:     $(date)"
echo ""

srun ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/scripts/qpy_export.py" \
    "${HPCQC_ROOT}/${RESULT_JSON}"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
