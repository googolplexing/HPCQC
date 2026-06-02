#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=wsa_diversity
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --output=slurm_logs/wsa_diversity.o%j
#SBATCH --error=slurm_logs/wsa_diversity.e%j
#
# Workstream A §7.3 — disjoint placement-equivalence (BLUE-PROPOSAL-WORKSTREAM-A;
# RED-RULING-WORKSTREAM-A §7.3).
#
# Runs the solver's `disjoint` selection on cal 08c3c70f (10q chain), then proves
# the selected set's multi-placement device-cal subtrees are byte-identical to the
# isolated single-placement runs — the F5a no-cross-talk oracle, now over
# SOLVER-SELECTED chains rather than hand-picked HIGH/LOW. Record-inventory guard
# runs first (vacuous-pass hole). Tiny scale (2 seeds x 6 kicks x 100 shots) — the
# no-cross-talk property is scale-independent.
#
# This launcher invokes the harness main() DIRECTLY (no pytest skip-gate): the
# placement-equivalence gate must run for real, never as a no-op skip.
#
# Usage: sbatch tests/slurm_wsa_diversity.sh
# Expected: WSA DIVERSITY EQUIVALENCE: ALL CHECKS PASSED   (exit 0)

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"
export HPCQC_RUN_SLOW=1

mkdir -p slurm_logs

echo "=== Workstream A — disjoint placement-equivalence (F5a oracle) ==="
srun ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
    python3 "${SLURM_SUBMIT_DIR}/tests/f5a_diversity_equivalence_validation.py"
