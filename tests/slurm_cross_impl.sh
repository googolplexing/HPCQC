#!/bin/bash
#SBATCH --job-name=cross_impl_val
#SBATCH --partition=standard-g
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/cross_impl_val.o%j
#SBATCH --error=slurm_logs/cross_impl_val.e%j

# Cross-implementation validation — RED-SPEC-001 §7.5
#
# Verifies the framework's exact ground state energy for TFIM 4q against:
#   1. Pure numpy tensor product construction (no Qiskit, no HPCQC)
#   2. Framework BYO Hamiltonian plugin (SparsePauliOp path)
#
# Both must agree with -4.75877048 to within 1e-6.
#
# Usage: sbatch tests/slurm_cross_impl.sh
# Expected: CROSS-IMPLEMENTATION VALIDATION: ALL PASS

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

echo "=== Cross-Implementation Validation — RED-SPEC-001 §7.5 ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_GPU_CONTAINER}"
echo "Started:   $(date)"
echo ""

srun ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/scripts/cross_impl_validation.py"

EXIT_CODE=$?

echo ""
echo "Finished: $(date)"
echo "Exit code: ${EXIT_CODE}"

exit ${EXIT_CODE}
