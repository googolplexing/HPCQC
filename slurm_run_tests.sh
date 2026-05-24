#!/bin/bash
# Run the F-6 regression + Gap A scaffold tests on a LUMI standard (CPU) node,
# inside the qiskit-aer container, via the same srun + wrapper pattern as
# slurm_floquet_40i_60k_1000s.sh. Submit from the repo root:
#     sbatch slurm_run_tests.sh
#SBATCH --job-name=byo_tests
#SBATCH --account=project_462000055
#SBATCH --partition=standard
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=slurm_logs/byo_tests.o%j
#SBATCH --error=slurm_logs/byo_tests.e%j

# env.sh defines HPCQC_ROOT, HPCQC_CPU_WRAPPER, HPCQC_CPU_CONTAINER
source "${SLURM_SUBMIT_DIR}/env.sh"
mkdir -p "${HPCQC_ROOT}/slurm_logs"

# Make the package importable inside the container (mirrors the floquet runner).
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"

echo "Container : ${HPCQC_CPU_CONTAINER}"
echo "HPCQC root: ${HPCQC_ROOT}"
echo "Started   : $(date)"
echo

RC_PYTEST=0
RC_SCAFFOLD=0

echo "=== F-6 regression (pytest; needs the real qiskit imports) ==="
srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 -m pytest "${HPCQC_ROOT}/tests/unit/test_noise_config_validation.py" -v \
    || RC_PYTEST=$?

echo
echo "=== Gap A scaffold (standalone script; stdlib + numpy) ==="
srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${HPCQC_ROOT}/tests/unit/test_byo_sweep.py" \
    || RC_SCAFFOLD=$?

echo
echo "=== Summary ==="
echo "  F-6 pytest      exit=${RC_PYTEST}"
echo "  Gap A scaffold  exit=${RC_SCAFFOLD}"
echo "Finished  : $(date)"

# Nonzero if either suite failed.
[ "${RC_PYTEST}" -eq 0 ] && [ "${RC_SCAFFOLD}" -eq 0 ]
exit $?
