#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# v1.2.0 Full Regression — submit all E-step test suites
#
# Run this on the LUMI login node (NOT via sbatch — this IS the launcher).
#
# Usage:
#   bash tests/run_all_tests.sh
#
# It submits 13 SLURM jobs in parallel, captures their job IDs,
# and prints a check command to run once all jobs finish.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════════"
echo "  v1.2.0 FULL REGRESSION — Submitting all test suites"
echo "  Date: $(date -Iseconds)"
echo "═══════════════════════════════════════════════════════════"
echo ""

mkdir -p slurm_logs

JOBS=""
NAMES=""

submit() {
    local script="$1"
    local name
    name=$(basename "$script" .sh)
    JID=$(sbatch --parsable "$script")
    echo "  ✓ $name → Job $JID"
    JOBS="$JOBS $JID"
    NAMES="$NAMES $name:$JID"
}

submit tests/slurm_e1.sh
submit tests/slurm_e2_planner.sh
submit tests/slurm_e2_stress.sh
submit tests/slurm_e3.sh
submit tests/slurm_e4.sh
submit tests/slurm_e5.sh
submit tests/slurm_e6a.sh
submit tests/slurm_e6b.sh
submit tests/slurm_e7.sh
submit tests/slurm_e8.sh
submit tests/slurm_e9.sh
submit tests/slurm_e10.sh
submit tests/test_v111_validate.sh

JOBS="${JOBS# }"  # trim leading space

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  13 jobs submitted"
echo "  Job IDs: $JOBS"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "Monitor progress:"
echo "  squeue --me"
echo ""

# Write a checker script with job IDs baked in
cat > check_results.sh << CHECKER
#!/bin/bash
PASS=0; FAIL=0
for JID in $JOBS; do
  OFILE=\$(ls slurm_logs/*.o\${JID} 2>/dev/null | head -1)
  EFILE=\$(ls slurm_logs/*.e\${JID} 2>/dev/null | head -1)
  if [ -z "\$OFILE" ]; then
    echo "[????] Job \$JID — no output file"
    FAIL=\$((FAIL+1))
  elif tail -20 "\$OFILE" | grep -q "ALL CHECKS PASSED\|ALL.*PASSED"; then
    NAME=\$(head -5 "\$OFILE" | grep -oP '(?<=  )[A-Za-z0-9_.+ -]+(?= Validation|VALIDATION)' | head -1)
    echo "[PASS] Job \$JID — \$NAME"
    echo "       stdout: \$OFILE"
    PASS=\$((PASS+1))
  else
    NAME=\$(head -5 "\$OFILE" | grep -oP '(?<=  )[A-Za-z0-9_.+ -]+(?= Validation|VALIDATION)' | head -1)
    echo "[FAIL] Job \$JID — \$NAME"
    echo "       stdout: \$OFILE"
    echo "       stderr: \$EFILE"
    echo "       last output: \$(tail -3 "\$OFILE" | head -1)"
    FAIL=\$((FAIL+1))
  fi
done
echo ""
echo "Total: \$PASS passed, \$FAIL failed out of \$((PASS+FAIL)) suites"
CHECKER
chmod +x check_results.sh

echo "Once squeue --me shows no jobs, run:"
echo "  bash check_results.sh"
