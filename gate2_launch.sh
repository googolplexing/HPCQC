#!/bin/bash
set -uo pipefail
BASE="${GATE2_BASE:-examples/byo/floquet_dtc_q10_sweep.yaml}"
OUTROOT="${GATE2_OUTROOT:-results/gate2}"
SEEDS="$*"
if [[ -z "$SEEDS" ]]; then echo "ERROR: no seeds given" >&2; exit 2; fi
mkdir -p "$OUTROOT/configs"
declare -a PIDS=() PSEEDS=()
for s in $SEEDS; do
  ss=$(printf '%02d' "$s"); cfg="$OUTROOT/configs/seed_${ss}.yaml"; outdir="$OUTROOT/seed_${ss}"
  python3 - "$BASE" "$s" "$outdir" > "$cfg" <<'PY'
import sys, yaml
base, seed, outdir = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(base) as f: d = yaml.safe_load(f)
exps = d["sweep"]["experiments"]; assert len(exps) == 1
exps[0]["seed_list"] = [seed]; d["sweep"]["output_dir"] = outdir
sys.stdout.write(yaml.safe_dump(d, sort_keys=False))
PY
  echo "[gate2] seed ${ss}: ${cfg} -> ${outdir}"
  OMP_NUM_THREADS=1 python3 -m lumi_hpc_qc.sweep.run_sweep "$cfg" > "$OUTROOT/seed_${ss}.log" 2>&1 &
  PIDS+=("$!"); PSEEDS+=("$ss")
done
echo "[gate2] launched ${#PIDS[@]} concurrent single-seed processes; waiting..."
FAIL=0
for i in "${!PIDS[@]}"; do
  if ! wait "${PIDS[$i]}"; then echo "[gate2] FAIL: seed ${PSEEDS[$i]} (pid ${PIDS[$i]})" >&2; FAIL=1; fi
done
if [[ "$FAIL" -ne 0 ]]; then echo "[gate2] one or more seeds FAILED — see $OUTROOT/seed_*.log" >&2; exit 1; fi
echo "[gate2] all ${#PIDS[@]} seeds completed OK"
