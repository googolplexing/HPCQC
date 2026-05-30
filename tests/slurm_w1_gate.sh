#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#SBATCH --job-name=w1_gate
#SBATCH --partition=standard
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --output=slurm_logs/w1_gate.o%j
#SBATCH --error=slurm_logs/w1_gate.e%j
#
# W1.6 — 40-seed production gate-2 reproduction, gated on the z_comb verifier.
#
# Runs the CANONICAL, UNMODIFIED examples/byo/floquet_dtc_q10_sweep.yaml (40
# seeds x 60 kicks x 2 arms, calibration 08c3c70f, opt_level 3 [default], 1000
# shots) through the W1 forkserver engine, then gates the device-calibrated
# aggregated autocorrelator against the runner reference via
# tests/_w1_z_comb_verify.py (PASS iff 0 kicks beyond 5 sigma; >3 sigma flagged
# for review per NF1). This is RED-RESP-W1-CAP-VERIFY-AND-GATE-RULING item 6 /
# RED-RESP-W1-GATE-CLOSURE §7 — the one remaining step for W1.6 sign-off.
#
# Why --output-dir instead of editing the YAML or duplicating it:
#   The production YAML is the single canonical config. We point it at a clean,
#   run-specific workdir via run_sweep's --output-dir override (in-memory only;
#   the file is never written back), so the gate and the production sweep share
#   ONE config with no drift surface. The override is unit-tested offline at
#   tests/unit/test_run_sweep_output_dir_override.py.
#
# Cap expectation (banked from the §3 sibling-map fix; capture the footer):
#   40 seeds x 2 arms = 80 units; usable_cores_physical=128 (post core-count
#   fix), so cap = min(128, 80, 128, mem_term~151) = 80, binding=num_units,
#   1 wave. The dispatch footer in this job's slurm-*.out should read
#   `cap=80; binding=num_units; 1 wave(s)` — bank it (carry-forward #3) so the
#   1-wave consequence is recorded on the production run, not only inferred.
#   Memory headroom: 80 x ~1.30 GiB = ~104 GiB << safe_mem ~197 GiB.
#
# Path discipline (mirrors tests/slurm_w1_canary.sh):
#   - cd $SLURM_SUBMIT_DIR (= repo root) so the YAML's repo-relative paths
#     (examples/byo/..., examples/q50_...) resolve CWD-relative.
#   - The engine writes under $OUTPUT_DIR (set via --output-dir). The
#     device-calibrated aggregated dat lands at
#       $OUTPUT_DIR/byo_dat/floquet_dtc/<phys-qubit-set>/device_calibrated/aggregated_autocorr.dat
#     One placement (top_1) -> exactly one such file; we glob and assert that.
#   - Pre-run `rm -rf "$OUTPUT_DIR"` guarantees a deterministic workdir and no
#     campaign_manifest resume that would skip seeds.
#
# Invocation idiom: srun <wrapper> <container> python3 path/to/script.py
#   (no -c, no heredoc through the wrapper).
#
# Usage: sbatch tests/slurm_w1_gate.sh
# Expected: verifier prints "VERDICT : PASS" and exits 0; set -e fails the job
#   on any non-zero (FAIL=1, STRUCTURAL=3).

set -euo pipefail
source "${SLURM_SUBMIT_DIR}/env.sh"

mkdir -p slurm_logs

GATE_YAML="${HPCQC_ROOT}/examples/byo/floquet_dtc_q10_sweep.yaml"
REFERENCE="${HPCQC_ROOT}/examples/reference/floquet_dtc_q10_autocorr.csv"
VERIFIER="${HPCQC_ROOT}/tests/_w1_z_comb_verify.py"
OUTPUT_DIR="${SLURM_SUBMIT_DIR}/sweep_output/w1_gate"
REPORT="${OUTPUT_DIR}/w1_gate_z_comb_report.csv"

echo "=== W1.6 — 40-seed production gate-2 (z_comb) ==="
echo "Job ID:    ${SLURM_JOB_ID}"
echo "Node:      $(hostname)"
echo "Container: ${HPCQC_CPU_CONTAINER}"
echo "YAML:      ${GATE_YAML} (unmodified; --output-dir override)"
echo "Reference: ${REFERENCE}"
echo "OutputDir: ${OUTPUT_DIR}"
echo "Verifier:  ${VERIFIER}"
echo "Started:   $(date)"
echo ""

# Deterministic workdir: no stale dats, no resume-mode seed skipping.
rm -rf "${OUTPUT_DIR}"

export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"
export SINGULARITYENV_PYTHONPATH="${HPCQC_ROOT}/src"

# ── (1) Sweep: canonical YAML, clean workdir via --output-dir. ──
cd "${SLURM_SUBMIT_DIR}"
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 -m lumi_hpc_qc.sweep.run_sweep "${GATE_YAML}" \
        --output-dir "${OUTPUT_DIR}"

echo ""
echo "=== Locate device-calibrated aggregated autocorrelator ==="

# One placement (top_1) -> exactly one device_calibrated aggregated dat.
# Glob and assert exactly one match before gating.
shopt -s nullglob
CANDIDATES=( "${OUTPUT_DIR}"/byo_dat/floquet_dtc/*/device_calibrated/aggregated_autocorr.dat )
shopt -u nullglob
if [ "${#CANDIDATES[@]}" -ne 1 ]; then
    echo "ERROR: expected exactly 1 device_calibrated aggregated_autocorr.dat under" >&2
    echo "       ${OUTPUT_DIR}/byo_dat/floquet_dtc/*/device_calibrated/, found ${#CANDIDATES[@]}:" >&2
    printf '       %s\n' "${CANDIDATES[@]}" >&2
    exit 2
fi
CANDIDATE="${CANDIDATES[0]}"
echo "Candidate: ${CANDIDATE}"
echo ""

echo "=== (2) z_comb gate (PASS iff 0 kicks beyond 5 sigma) ==="

# Stdlib-only verifier; run it through the same wrapper for consistency.
# set -e fails the job on verifier exit 1 (gate FAIL) or 3 (STRUCTURAL).
srun "${HPCQC_CPU_WRAPPER}" "${HPCQC_CPU_CONTAINER}" \
    python3 "${VERIFIER}" \
        --candidate "${CANDIDATE}" \
        --reference "${REFERENCE}" \
        --candidate-seeds 40 \
        --reference-seeds 40 \
        --report "${REPORT}"

echo ""
echo "Gate PASSED. Bank for W1.6 sign-off:"
echo "  - this slurm log (carries the 'cap=80; binding=num_units; 1 wave(s)' footer)"
echo "  - ${CANDIDATE}"
echo "  - ${REPORT}"
echo "  - the sacct accounting row"
echo "Finished: $(date)"
