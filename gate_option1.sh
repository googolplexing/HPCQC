#!/bin/bash
# W1.6 Option-1 gate launcher (RED ruling #3 / fold §4.1).
#
# Runs the pinned-path gate end to end, inside the container, as a SINGLE
# engine run_sweep -- the W1 engine does its own internal seed parallelism, so
# this does NOT use the fork-per-seed gate2_launch.sh (that workaround OOM'd at
# the 40-seed scale; see RED-RESP-W1-PARALLELISM-AND-OOM-ROOTCAUSE-v1.4).
#
# Both arms are pinned to the canonical placement via the config's
# physical_qubits field; the device-calibrated arm is gated against the banked
# pinned reference with the PURE z_comb gate (default mode=gate, 5-sigma; no
# --floor / --max-rel-dev / --mode step1-residual).
#
# Submit (cd HPCQC root on LUMI first):
#   sbatch --account=project_462001289 --partition=standard --nodes=1 \
#     --ntasks=1 --cpus-per-task=256 --mem=0 --time=01:00:00 \
#     --job-name=gate_option1 \
#     --output=slurm_logs/gate_option1.o%j --error=slurm_logs/gate_option1.e%j \
#     --wrap 'cd "$SLURM_SUBMIT_DIR" && source "$SLURM_SUBMIT_DIR/env.sh" && \
#       export SINGULARITYENV_PYTHONPATH="$HPCQC_ROOT/src" && \
#       srun $HPCQC_CPU_WRAPPER $HPCQC_CPU_CONTAINER bash gate_option1.sh'
#
# Overridable via env: GATE_CFG, GATE_DISORDER, GATE_REF, GATE_OUTDIR, GATE_NSEEDS.
set -uo pipefail

CFG="${GATE_CFG:-examples/byo/floquet_dtc_q10_sweep.yaml}"
DISORDER="${GATE_DISORDER:-examples/byo/floquet_disorder_q10.json}"
REF="${GATE_REF:-examples/reference/floquet_dtc_q10_autocorr.csv}"
OUTDIR="${GATE_OUTDIR:-results/gate_option1_${SLURM_JOB_ID:-local}}"
NSEEDS="${GATE_NSEEDS:-40}"

# Avoid BLAS oversubscription when the engine forks per-seed workers.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"

echo "[gate] cfg=$CFG disorder=$DISORDER ref=$REF outdir=$OUTDIR nseeds=$NSEEDS"

# ── 1. Pre-flight: master_seed==0 + pinned placement + file-disorder (fast) ──
echo "[gate] pre-flight ..."
python3 tests/_w1_gate_preflight.py "$CFG" "$DISORDER" || {
  echo "[gate] PRE-FLIGHT FAILED -- not launching the sweep" >&2; exit 2; }

# ── 2. Canonical drift / YAML-pin guard (solver still emits the recorded top_1
#       AND the YAML pins exactly it). Single _CANONICAL authority. Requires
#       pytest + rustworkx (the unit-suite container). If pytest is absent,
#       skip with a loud warning (step 1 already enforces *a* pinned placement,
#       and the identity guard runs in the regular unit suite); force-skip with
#       GATE_SKIP_GUARD=1. ──
if [[ "${GATE_SKIP_GUARD:-0}" == "1" ]]; then
  echo "[gate] WARNING: canonical guard SKIPPED (GATE_SKIP_GUARD=1) -- confirm it passed in the unit suite" >&2
elif ! python3 -c "import pytest" 2>/dev/null; then
  echo "[gate] WARNING: pytest unavailable -- canonical guard SKIPPED; run tests/unit/test_canonical_placement_guard.py in the unit suite" >&2
else
  echo "[gate] canonical placement guard ..."
  python3 -m pytest -q tests/unit/test_canonical_placement_guard.py || {
    echo "[gate] CANONICAL GUARD FAILED -- placement drifted or YAML un-pinned" >&2
    exit 2; }
fi

# ── 3. Single engine run (internal parallelism over the 40 seeds) ──
echo "[gate] run_sweep -> $OUTDIR ..."
python3 -m lumi_hpc_qc.sweep.run_sweep "$CFG" --output-dir "$OUTDIR" || {
  echo "[gate] run_sweep FAILED" >&2; exit 1; }

# ── 4. Locate the device-calibrated candidate aggregate. Pinned => exactly one
#       placement subdir under the device_calibrated env. ──
shopt -s nullglob
CANDS=( "$OUTDIR"/byo_dat/floquet_dtc/*/device_calibrated/aggregated_autocorr.dat )
if [[ ${#CANDS[@]} -ne 1 ]]; then
  echo "[gate] expected exactly 1 device_calibrated candidate aggregate, found ${#CANDS[@]}:" >&2
  printf '  %s\n' "${CANDS[@]:-<none>}" >&2
  exit 3
fi
CAND="${CANDS[0]}"
echo "[gate] candidate: $CAND"

# ── 5. Pure z_comb gate (default mode=gate, 5-sigma; no floor/max-rel-dev) ──
echo "[gate] z_comb verify (pure gate mode) vs $REF ..."
python3 tests/_w1_z_comb_verify.py \
  --candidate "$CAND" \
  --reference "$REF" \
  --candidate-seeds "$NSEEDS" \
  --reference-seeds "$NSEEDS" \
  --report "$OUTDIR/z_comb_report.csv"
RC=$?

echo "[gate] verifier exit=$RC (0=PASS  1=FAIL  3=STRUCTURAL)"
echo "[gate] report: $OUTDIR/z_comb_report.csv"
exit $RC
