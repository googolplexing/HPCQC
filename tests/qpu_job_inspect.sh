#!/bin/bash
#SBATCH --job-name=qpu_inspect
#SBATCH --account=project_462001126
#SBATCH --partition=q_fiqci
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/qpu_inspect.o%j
#SBATCH --error=slurm_logs/qpu_inspect.e%j

module use /appl/local/quantum/modulefiles
module load fiqci-vtt-qiskit
export DEVICES=("Q50")
source $RUN_SETUP

# Pass optional job_id as argument: sbatch tests/qpu_job_inspect.sh <job_id>
# Without argument, inspects all 4 completed jobs from the timing benchmark
python tests/qpu_job_inspect.py $1
