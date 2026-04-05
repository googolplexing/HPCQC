#!/bin/bash
#SBATCH --job-name=q50_cal_fetch
#SBATCH --account=project_462001126
#SBATCH --partition=q_fiqci
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/q50_cal_fetch.o%j
#SBATCH --error=slurm_logs/q50_cal_fetch.e%j

module use /appl/local/quantum/modulefiles
module load fiqci-vtt-qiskit
export DEVICES=("Q50")
source $RUN_SETUP

# Fetch latest calibration (or pass a specific calibration_set_id)
python tests/fetch_q50_calibration.py $1
