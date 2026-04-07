#!/bin/bash
#SBATCH --job-name=qpu_timing_query
#SBATCH --account=project_462001126
#SBATCH --partition=q_fiqci
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/qpu_timing_query.o%j
#SBATCH --error=slurm_logs/qpu_timing_query.e%j

module use /appl/local/quantum/modulefiles
module load fiqci-vtt-qiskit
export DEVICES=("Q50")
source $RUN_SETUP

# Query timing breakdown for completed jobs.
# No circuit submission. No QPU cost. Read-only.
#
# Default: queries 5 known timing benchmark jobs from April 7.
# Or pass specific job IDs: sbatch tests/qpu_timing_query.sh <job_id_1> <job_id_2>
python tests/qpu_timing_query.py $@
