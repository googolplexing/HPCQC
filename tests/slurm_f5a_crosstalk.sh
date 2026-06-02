#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=f5a_crosstalk
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:10:00
#SBATCH --output=slurm_logs/f5a_crosstalk.o%j
#SBATCH --error=slurm_logs/f5a_crosstalk.e%j
#
# F5a no-cross-talk CI integration test
# (BLUE-PROPOSAL-F5A-NO-CROSSTALK-CI; RED-RULING-F5A-NO-CROSSTALK-CI).
#
# Promotes the Piece-3 hand diff to a standing guard: multi-placement device-cal
# subtrees byte-identical to the isolated single-placement runs (no cross-talk),
# noiseless byte-identical across placements (control), flag False(multi)/True(single).
# Tiny scale (2 seeds x 6 kicks x 100 shots) — the invariant is scale-independent.
#
# This launcher invokes the harness main() DIRECTLY — it does NOT go through the
# pytest HPCQC_RUN_SLOW skip-gate. That is deliberate: the real CI gate before the
# VIP banks results must run for real, never as a no-op skip. (The pytest wrapper's
# gate exists only to keep `pytest tests/unit` fast/discoverable; the actual gate
# is this job.) HPCQC_RUN_SLOW is also exported for any nested gate, belt-and-braces.
#
# Usage: sbatch tests/slurm_f5a_crosstalk.sh
# Expected: F5A NO-CROSSTALK: ALL CHECKS PASSED   (exit 0)

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export HPCQC_RUN_SLOW=1

mkdir -p slurm_logs

echo "=== F5a No-Cross-Talk CI Integration Test ==="
srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${SLURM_SUBMIT_DIR}/tests/f5a_no_crosstalk_validation.py"
