#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# Diagnostic: run ONE instance of floquet_runner.py with everything
# unbuffered so the actual Python error surfaces in the log. Also runs
# a pre-flight container/import sanity check before the real call.
#
# Usage:
#   sbatch slurm_floquet_debug.sh
#
#SBATCH --job-name=floquet_dbg
#SBATCH --account=project_462000055
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --time=00:05:00
#SBATCH --output=slurm_logs/floquet_dbg.o%j
#SBATCH --error=slurm_logs/floquet_dbg.e%j

set -uo pipefail   # NOT -e: we want to see what fails, not abort early
set -x             # trace every command in the log

export LUMI_QISKIT_SINGULARITY_CONTAINER_PATH="${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH:-/appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif}"
export WRAPPER_PATH="${WRAPPER_PATH:-/appl/local/quantum/qiskit/run-singularity}"

HPCQC_ROOT="${HPCQC_ROOT:-${SLURM_SUBMIT_DIR}}"
mkdir -p "${HPCQC_ROOT}/slurm_logs" "${HPCQC_ROOT}/results/floquet_debug_job${SLURM_JOB_ID}"

echo "================================================================"
echo " Floquet single-instance DEBUG run"
echo "================================================================"
echo "Job ID     : ${SLURM_JOB_ID}"
echo "Node       : $(hostname)"
echo "HPCQC root : ${HPCQC_ROOT}"
echo "Container  : ${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH}"
echo "Wrapper    : ${WRAPPER_PATH}"
echo "Started    : $(date)"
echo

# ── env to propagate into the container ──
export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export SINGULARITYENV_ROCR_VISIBLE_DEVICES=""
export SINGULARITYENV_HSA_TOOLS_LIB=""
export SINGULARITYENV_PYTHONUNBUFFERED=1   # ← the key fix
export SINGULARITYENV_PYTHONFAULTHANDLER=1 # ← native traceback on C-level crash
export SINGULARITYENV_OMP_NUM_THREADS=1
export SINGULARITYENV_OPENBLAS_NUM_THREADS=1
export SINGULARITYENV_MKL_NUM_THREADS=1

# ─────────────────────────────────────────────────────────────────
echo
echo "═══ STEP 1: container + qiskit-aer import sanity check ═══"
echo
${WRAPPER_PATH} ${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH} \
    python3 -u -c '
import sys, platform
print("python    :", sys.version)
print("platform  :", platform.platform())
import qiskit
print("qiskit    :", qiskit.__version__)
import qiskit_aer
print("qiskit-aer:", qiskit_aer.__version__)
from qiskit_aer import AerSimulator
sim = AerSimulator()
print("AerSimulator() built OK ->", sim)
from qiskit import QuantumCircuit
qc = QuantumCircuit(2, 2); qc.h(0); qc.cx(0,1); qc.measure([0,1],[0,1])
from qiskit.compiler import transpile
tqc = transpile(qc, sim, optimization_level=3)
res = sim.run(tqc, shots=128).result()
print("toy circuit counts:", res.get_counts())
print("STEP 1 PASS")
'
STEP1_RC=$?
echo "STEP 1 exit code: ${STEP1_RC}"

# ─────────────────────────────────────────────────────────────────
echo
echo "═══ STEP 2: file visibility check inside container ═══"
echo
${WRAPPER_PATH} ${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH} \
    bash -c "ls -l '${HPCQC_ROOT}/floquet_runner.py' && \
             ls -d '${HPCQC_ROOT}/src/lumi_hpc_qc/backends' && \
             echo STEP 2 PASS"
STEP2_RC=$?
echo "STEP 2 exit code: ${STEP2_RC}"

# ─────────────────────────────────────────────────────────────────
echo
echo "═══ STEP 3: run a single floquet instance (noiseless, unbuffered) ═══"
echo
${WRAPPER_PATH} ${LUMI_QISKIT_SINGULARITY_CONTAINER_PATH} \
    python3 -u "${HPCQC_ROOT}/floquet_runner.py" \
        --instance-id 0 \
        --backend noiseless \
        --output-dir "${HPCQC_ROOT}/results/floquet_debug_job${SLURM_JOB_ID}"
STEP3_RC=$?
echo "STEP 3 exit code: ${STEP3_RC}"

echo
echo "================================================================"
echo " Finished : $(date)"
echo " STEP 1 (import sanity)        : ${STEP1_RC}"
echo " STEP 2 (file visibility)      : ${STEP2_RC}"
echo " STEP 3 (one noiseless instance): ${STEP3_RC}"
echo "================================================================"
