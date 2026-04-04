#!/bin/bash
#SBATCH --job-name=qpu_timing
#SBATCH --account=project_462000888
#SBATCH --partition=q_fiqci
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2G
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/qpu_timing.o%j
#SBATCH --error=slurm_logs/qpu_timing.e%j

module use /appl/local/quantum/modulefiles
module load fiqci-vtt-qiskit
export DEVICES=("Q50")
source $RUN_SETUP

python tests/qpu_simple_timing.py
