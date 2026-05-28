#!/usr/bin/env python3
"""Gate-2 combine + per-kick comparison (the §6/§7 workaround's analysis step).

Gathers the per-seed device_calibrated autocorrelator vectors written by the
concurrent single-seed runs (gate2_launch.sh), asserts the expected number of
seeds are present BEFORE aggregating (a silently-missing seed would shift the
mean — RED-RESP-D3.4C F2 discipline), aggregates them into a mean + per-kick sem
(sem = sample_stdev(ddof=1)/sqrt(N), matching aggregate_floquet.py /
aggregate_byo_autocorr), and compares per kick against the banked reference CSV's
device_cal columns.

Pure stdlib (no numpy/qiskit) so it runs on a login node or in the container.

Per-seed input (one per seed dir, top_1 device_calibrated guardrail = single
placement):
  <outroot>/seed_NN/byo_dat/<stem>/<phys>/device_calibrated/instance_NN_autocorr.dat
    format: "# kick   autocorrelator" then "{kick:4d} {autocorr:10.4f}"

Reference CSV columns: kick,noiseless_mean,noiseless_sem,device_cal_mean,device_cal_sem

Outputs:
  <out>/aggregated_autocorr.dat       "# kick  mean_autocorr  sem"
  <out>/gate2_compare.csv             kick, combined_mean, combined_sem,
                                      ref_mean, ref_sem, delta, z_ref, z_combined
  stdout summary (max |delta|, worst z, n kicks beyond thresholds)

Exit codes: 0 = ran + compared (pass/fail is Red's call on the evidence);
            2 = seed-coverage assertion failed (fewer than --expect-seeds);
            3 = structural problem (no dats, ragged vectors, kick-grid mismatch).
"""
import argparse
import csv
import glob
import math
import os
import re
import statistics
import sys


def _load_instance_dat(path):
    """Return list[float] autocorrelator column from an instance_NN dat."""
    vals = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            vals.append(float(parts[1]))
    return vals


def gather_per_seed(outroot):
    """Return dict seed -> autocorr vector for the device_calibrated arm."""
    pattern = os.path.join(
        outroot, "seed_*", "byo_dat", "*", "*", "device_calibrated",
        "instance_*_autocorr.dat",
    )
    found = {}
    for p in sorted(glob.glob(pattern)):
        m = re.search(r"instance_(\d+)_autocorr\.dat$", os.path.basename(p))
        if not m:
            continue
        seed = int(m.group(1))
        if seed in found:
            print(f"[gate2-compare] STRUCTURAL: duplicate device_calibrated "
                  f"dat for seed {seed}: {p} and {found[seed][1]}", file=sys.stderr)
            sys.exit(3)
        found[seed] = (_load_instance_dat(p), p)
    return found


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gate-2 combine + per-kick compare.")
    ap.add_argument("--outroot", default="results/gate2",
                    help="dir containing seed_NN/ subdirs (default results/gate2)")
    ap.add_argument("--reference",
                    default="examples/reference/floquet_dtc_q10_autocorr.csv")
    ap.add_argument("--expect-seeds", type=int, default=40,
                    help="required seed count; fail loud if fewer (default 40)")
    ap.add_argument("--out", default=None,
                    help="output dir (default <outroot>/combined)")
    args = ap.parse_args(argv)
    out = args.out or os.path.join(args.outroot, "combined")
    os.makedirs(out, exist_ok=True)

    # ── gather + seed-coverage assertion (BEFORE aggregating) ──
    found = gather_per_seed(args.outroot)
    if not found:
        print(f"[gate2-compare] STRUCTURAL: no device_calibrated instance dats "
              f"under {args.outroot}/seed_*/byo_dat/", file=sys.stderr)
        sys.exit(3)
    seeds = sorted(found)
    if len(seeds) < args.expect_seeds:
        missing = sorted(set(range(args.expect_seeds)) - set(seeds))
        print(f"[gate2-compare] SEED COVERAGE FAIL: found {len(seeds)} of "
              f"{args.expect_seeds} seeds; missing {missing}. Refusing to "
              f"aggregate a short ensemble (would shift the mean).", file=sys.stderr)
        sys.exit(2)
    if len(seeds) > args.expect_seeds:
        print(f"[gate2-compare] WARNING: found {len(seeds)} seeds > expected "
              f"{args.expect_seeds}; aggregating all of them.", file=sys.stderr)

    # ── shape check ──
    lengths = {len(v) for v, _ in found.values()}
    if len(lengths) != 1:
        print(f"[gate2-compare] STRUCTURAL: ragged autocorr vectors across "
              f"seeds (lengths {sorted(lengths)}).", file=sys.stderr)
        sys.exit(3)
    n_kicks = lengths.pop()

    # ── aggregate: mean + sem(ddof=1)/sqrt(N) per kick ──
    n = len(seeds)
    mean = [0.0] * n_kicks
    sem = [0.0] * n_kicks
    for k in range(n_kicks):
        col = [found[s][0][k] for s in seeds]
        mean[k] = sum(col) / n
        sem[k] = (statistics.stdev(col) / math.sqrt(n)) if n > 1 else 0.0

    agg_path = os.path.join(out, "aggregated_autocorr.dat")
    with open(agg_path, "w", encoding="utf-8") as f:
        f.write("# kick  mean_autocorr  sem\n")
        for k in range(n_kicks):
            f.write(f"{k:4d} {mean[k]:10.4f} {sem[k]:10.4f}\n")

    # ── load reference device_cal columns ──
    ref_mean, ref_sem = {}, {}
    with open(args.reference, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = int(row["kick"])
            ref_mean[k] = float(row["device_cal_mean"])
            ref_sem[k] = float(row["device_cal_sem"])

    # ── per-kick compare ──
    cmp_path = os.path.join(out, "gate2_compare.csv")
    worst_abs = 0.0
    worst_z_ref = 0.0
    worst_z_comb = 0.0
    n_beyond_1ref = 0
    n_beyond_3comb = 0
    n_beyond_5comb = 0
    compared = 0
    with open(cmp_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kick", "combined_mean", "combined_sem",
                    "ref_mean", "ref_sem", "delta", "z_ref", "z_combined"])
        for k in range(n_kicks):
            if k not in ref_mean:
                continue
            compared += 1
            delta = mean[k] - ref_mean[k]
            z_ref = abs(delta) / ref_sem[k] if ref_sem[k] > 0 else float("inf")
            denom = math.sqrt(ref_sem[k] ** 2 + sem[k] ** 2)
            z_comb = abs(delta) / denom if denom > 0 else float("inf")
            w.writerow([k, f"{mean[k]:.4f}", f"{sem[k]:.4f}",
                        f"{ref_mean[k]:.4f}", f"{ref_sem[k]:.4f}",
                        f"{delta:+.4f}", f"{z_ref:.2f}", f"{z_comb:.2f}"])
            worst_abs = max(worst_abs, abs(delta))
            if math.isfinite(z_ref):
                worst_z_ref = max(worst_z_ref, z_ref)
            if math.isfinite(z_comb):
                worst_z_comb = max(worst_z_comb, z_comb)
            if z_ref > 1.0:
                n_beyond_1ref += 1
            if z_comb > 3.0:
                n_beyond_3comb += 1
            if z_comb > 5.0:
                n_beyond_5comb += 1

    print("=" * 64)
    print(f"  GATE-2 COMBINE + COMPARE  ({n} seeds, {n_kicks} kicks)")
    print("=" * 64)
    print(f"  seeds aggregated : {seeds[0]}..{seeds[-1]} ({n} of "
          f"{args.expect_seeds})")
    print(f"  aggregated .dat  : {agg_path}")
    print(f"  per-kick compare : {cmp_path}")
    print(f"  kicks compared   : {compared}")
    print(f"  max |delta|      : {worst_abs:.4f}")
    print(f"  worst z vs ref-sem        : {worst_z_ref:.2f}  "
          f"({n_beyond_1ref} kicks > 1.0)")
    print(f"  worst z vs combined-sigma : {worst_z_comb:.2f}  "
          f"({n_beyond_3comb} kicks > 3.0, {n_beyond_5comb} > 5.0)")
    print("=" * 64)
    if n_beyond_5comb:
        print("  NOTE: kicks beyond 5 combined-sigma — likely a real discrepancy,"
              " not statistical. Inspect before declaring pass.", file=sys.stderr)
    print("  (Pass criterion is Red's call on this evidence; this script reports,"
          " it does not self-certify.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
