#!/bin/bash
#SBATCH --job-name=fork_test
#SBATCH --partition=standard
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --output=slurm_logs/fork_test.o%j
#SBATCH --error=slurm_logs/fork_test.e%j

source "${SLURM_SUBMIT_DIR}/env.sh"

export SINGULARITYENV_OMP_NUM_THREADS=1

srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} python ${HPCQC_ROOT}/tests/fork_test_spawn.py
