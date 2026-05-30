#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""W1.6 z_comb gate verifier — pass/fail on the 40-seed device-calibrated run.

Unlike ``gate2_combine_compare.py`` (which *reports* and leaves the verdict to
Red), this artifact *gates*: it returns a pass/fail exit code so the 40-seed run
can be conditioned on it (RED-RESP-W1-CAP-VERIFY-AND-GATE-RULING §8 item 5; NF1).

It reads the W1-engine's already-aggregated output (``aggregated_autocorr.dat``,
written directly by ``aggregate_byo_autocorr`` — NOT re-gathered from per-seed
dats, which is the §6/§7 workaround path) and compares it per kick to the banked
runner reference's device_cal columns. The W1-engine and runner pipelines are
D9-disjoint (NF1), so this is a z_comb agreement test, NOT an aggregate
byte-match.

Per-kick combined-sigma z (identical formula to gate2_combine_compare.py):

    z_comb(k) = |cand_mean(k) - ref_mean(k)| / sqrt(ref_sem(k)^2 + cand_sem(k)^2)

Verdict (NF1 / §8):
    PASS  iff 0 kicks exceed 5.0 sigma.
    Kicks beyond 3.0 sigma are FLAGGED for investigation but do NOT fail.

The combined-sigma denominator is correct even when the reference and candidate
were aggregated over different seed counts (it is a combined uncertainty, not an
equal-N assumption); the seed counts are reported so a reviewer can see this.

Pure stdlib (no numpy/qiskit) so it runs on a login node or in the container.

Exit codes:
    0  PASS (0 kicks > 5 sigma).
    1  FAIL (>=1 kick > 5 sigma) — likely a real discrepancy, not statistical.
    3  STRUCTURAL (missing/unreadable input, kick-grid mismatch, no overlap).
"""
import argparse
import csv
import math
import sys

SIGMA_FAIL = 5.0
SIGMA_FLAG = 3.0


def load_aggregated_dat(path):
    """Read ``# kick  mean_autocorr  sem`` -> dict kick -> (mean, sem).

    Matches the format written by aggregate_byo_autocorr and
    gate2_combine_compare.py: comment lines start with '#', data lines are
    ``kick mean sem`` whitespace-separated.
    """
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(
                    f"{path}: expected 'kick mean sem', got {line!r}")
            k = int(parts[0])
            out[k] = (float(parts[1]), float(parts[2]))
    return out


def load_reference(path):
    """Read the runner reference CSV -> dict kick -> (device_cal_mean, sem)."""
    ref_mean, ref_sem = {}, {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            k = int(row["kick"])
            ref_mean[k] = float(row["device_cal_mean"])
            ref_sem[k] = float(row["device_cal_sem"])
    return ref_mean, ref_sem


def z_combined(cand_mean, cand_sem, ref_mean, ref_sem):
    """Per-kick combined-sigma z. Identical formula to gate2_combine_compare.py.

    Returns +inf when both sems are zero and the means differ (a real gap with
    no stated uncertainty -> cannot be explained as statistical).
    """
    delta = abs(cand_mean - ref_mean)
    denom = math.sqrt(ref_sem ** 2 + cand_sem ** 2)
    if denom > 0:
        return delta / denom
    return 0.0 if delta == 0 else float("inf")


def verify(candidate, ref_mean, ref_sem):
    """Compute per-kick z and the verdict. Pure; no I/O. Returns a dict.

    ``candidate`` is dict kick -> (mean, sem). Compares on the intersection of
    candidate and reference kicks. Raises ValueError if there is no overlap.
    """
    kicks = sorted(set(candidate) & set(ref_mean))
    if not kicks:
        raise ValueError("no overlapping kicks between candidate and reference")
    rows = []
    worst_z = 0.0
    n_flag = 0
    n_fail = 0
    for k in kicks:
        cm, cs = candidate[k]
        z = z_combined(cm, cs, ref_mean[k], ref_sem[k])
        rows.append((k, cm, cs, ref_mean[k], ref_sem[k], z))
        if math.isfinite(z):
            worst_z = max(worst_z, z)
        else:
            worst_z = float("inf")
        if z > SIGMA_FLAG:
            n_flag += 1
        if z > SIGMA_FAIL:
            n_fail += 1
    return {
        "rows": rows,
        "n_compared": len(kicks),
        "worst_z": worst_z,
        "n_flag": n_flag,        # > 3 sigma (includes the > 5 ones)
        "n_fail": n_fail,        # > 5 sigma
        "passed": n_fail == 0,
    }


# ── Step-1 systematic-residual reporting (RED-CLARIFICATION-STEP1-SIGMA-SYS) ──
# Step 1 is a MEASUREMENT run, not a 5-sigma gate. On identical qubits (the
# runner's self-selected set) it measures the cross-implementation residual
# between the runner's ALAP self-scheduling and the sweep's
# PadDelay+RelaxationNoisePass idle model. That residual BECOMES sigma_sys for
# the later Option-1 gate. The convergence call is made against a
# PRE-COMMITTED ceiling (a relative-deviation bound), which is a DIFFERENT
# quantity from sigma_sys: the ceiling is the defect line, fixed before the
# data; sigma_sys is the measured residual, known only after. ceiling >=
# sigma_sys. The statistical z_combined view (above) is the wrong bar here —
# a sub-percent systematic offset reads as tens of sigma against shot/seed
# sems regardless of correctness — which is why this mode is relative, not
# sem-normalized.

STEP1_FLOOR = 0.02          # |autocorr| noise floor; below it rel-dev is meaningless
STEP1_MAX_REL_DEV = 0.02    # pre-committed convergence ceiling (2%); Red approves first
STEP1_TREND_WARN = 0.0      # warn if rel-dev grows with depth (slope > this)


def relative_deviation(cand_mean, ref_mean, floor):
    """|cand - ref| / max(|ref|, floor).

    The floor guards the near-zero (tail / odd "non-return") kicks where
    |ref| -> 0 and a relative metric would otherwise blow up on noise.
    """
    return abs(cand_mean - ref_mean) / max(abs(ref_mean), floor)


def _ols_slope(xs, ys):
    """OLS slope of ys vs xs (pure stdlib). None if < 2 points or no x-variance."""
    n = len(xs)
    if n < 2:
        return None
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx


def fit_decay_rate(series, floor):
    """Decay rate b of |A(k)| ~ exp(b*k), via OLS on log|A| over the EVEN
    ('return') kicks above the floor.

    Even kicks carry the period-2 DTC echo envelope; odd kicks sit near zero
    (the runner finding: even/return kicks are the signal, odd kicks relax to
    ~0), so the decay is fit on the even-kick envelope only. Returns b
    (negative for decay) or None if < 3 usable points.
    """
    xs, ys = [], []
    for k in sorted(series):
        if k % 2 != 0:
            continue
        a = abs(series[k])
        if a < floor:
            continue
        xs.append(k)
        ys.append(math.log(a))
    if len(xs) < 3:
        return None
    return _ols_slope(xs, ys)


def residual_report(candidate, ref_mean, ref_sem, floor):
    """Per-kick systematic-residual view for Step 1. Pure; no I/O.

    ``candidate`` is dict kick -> (mean, sem); compared on the kick
    intersection. Returns per-kick rows plus the measured-residual summary
    (the sigma_sys candidates) and the decay-rate diagnostic. ``ref_sem`` is
    accepted for signature parity (not used: this view is deliberately NOT
    sem-normalized). Raises ValueError if there is no overlap.
    """
    kicks = sorted(set(candidate) & set(ref_mean))
    if not kicks:
        raise ValueError("no overlapping kicks between candidate and reference")
    rows = []
    rel_above = []          # [(kick, rel_dev)] for above-floor kicks
    signed = []             # signed (cand - ref), for the systematic direction
    cand_series = {}
    for k in kicks:
        cm, _cs = candidate[k]
        rm = ref_mean[k]
        cand_series[k] = cm
        above = abs(rm) >= floor
        rd = relative_deviation(cm, rm, floor)
        rows.append((k, cm, rm, abs(cm - rm), rd, above))
        signed.append(cm - rm)
        if above:
            rel_above.append((k, rd))
    max_rel = max((rd for _, rd in rel_above), default=0.0)
    max_rel_kick = max(rel_above, key=lambda t: t[1])[0] if rel_above else None
    mean_rel = (sum(rd for _, rd in rel_above) / len(rel_above)) if rel_above else 0.0
    trend = _ols_slope([k for k, _ in rel_above], [rd for _, rd in rel_above])
    b_cand = fit_decay_rate(cand_series, floor)
    b_ref = fit_decay_rate(ref_mean, floor)
    decay_rel = (abs(b_cand - b_ref) / abs(b_ref)
                 if (b_cand is not None and b_ref not in (None, 0)) else None)
    return {
        "rows": rows,
        "n_compared": len(kicks),
        "n_above_floor": len(rel_above),
        "max_rel_dev": max_rel,
        "max_rel_dev_kick": max_rel_kick,
        "mean_rel_dev": mean_rel,
        "mean_signed_residual": sum(signed) / len(signed),
        "rel_dev_trend_slope": trend,       # > 0 => growing with depth (compounding)
        "decay_rate_cand": b_cand,
        "decay_rate_ref": b_ref,
        "decay_rate_rel_diff": decay_rel,
    }


def _run_step1_residual(args, candidate, ref_mean, ref_sem):
    """Step-1 measurement: report the residual and call convergence vs the
    PRE-COMMITTED ceiling (args.max_rel_dev). Returns an exit code."""
    try:
        r = residual_report(candidate, ref_mean, ref_sem, args.floor)
    except ValueError as e:
        print(f"[step1-residual] STRUCTURAL: {e}", file=sys.stderr)
        return 3
    converged = r["max_rel_dev"] <= args.max_rel_dev

    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kick", "cand_mean", "ref_mean", "abs_resid",
                        "rel_dev", "above_floor"])
            for k, cm, rm, ad, rd, ab in r["rows"]:
                w.writerow([k, f"{cm:.6f}", f"{rm:.6f}", f"{ad:.6f}",
                            f"{rd:.4f}", int(ab)])

    def _f(x, p="{:.4e}"):
        return "n/a" if x is None else p.format(x)

    drd = r["decay_rate_rel_diff"]
    direction = ">" if r["mean_signed_residual"] > 0 else "<"
    print("=" * 64)
    print("  W1.6 STEP-1 RESIDUAL REPORT (measurement, not a 5-sigma gate)")
    print("=" * 64)
    print(f"  candidate        : {args.candidate}")
    print(f"  reference        : {args.reference}")
    note = _count_seeds_note(args)
    if note:
        print(note)
    print(f"  floor            : {args.floor:.4f}  (|ref| below it excluded)")
    print(f"  kicks compared   : {r['n_compared']}  ({r['n_above_floor']} above floor)")
    print(f"  max rel-dev      : {r['max_rel_dev'] * 100:.2f}%  "
          f"(kick {r['max_rel_dev_kick']})")
    print(f"  mean rel-dev     : {r['mean_rel_dev'] * 100:.2f}%")
    print(f"  mean signed resid: {r['mean_signed_residual']:+.6f}  "
          f"(systematic: cand {direction} ref)")
    print(f"  rel-dev trend    : {_f(r['rel_dev_trend_slope'], '{:.2e}')} /kick  "
          f"(>0 => compounding with depth)")
    print(f"  decay rate cand  : {_f(r['decay_rate_cand'])} /kick")
    print(f"  decay rate ref   : {_f(r['decay_rate_ref'])} /kick")
    print(f"  decay rate rdiff : "
          f"{('%.2f%%' % (drd * 100)) if drd is not None else 'n/a'}")
    print("-" * 64)
    print(f"  CEILING          : max rel-dev <= {args.max_rel_dev * 100:.2f}% "
          f"(pre-committed; != sigma_sys)")
    print(f"  VERDICT          : {'CONVERGED' if converged else 'NOT CONVERGED'}")
    if converged:
        print(f"  -> sigma_sys (measured) = {r['max_rel_dev'] * 100:.2f}% max / "
              f"{r['mean_rel_dev'] * 100:.2f}% mean rel-dev; carry to Option-1 gate.")
    else:
        print("  -> residual EXCEEDS the pre-committed ceiling: investigate the "
              "idle-decoherence implementation difference; do NOT widen the "
              "ceiling (RED-CLARIFICATION §5).", file=sys.stderr)
    if (r["rel_dev_trend_slope"] is not None
            and r["rel_dev_trend_slope"] > args.trend_warn):
        print("  NOTE: rel-dev grows with depth (positive trend) — inspect for "
              "compounding even under the magnitude ceiling.", file=sys.stderr)
    print("=" * 64)
    return 0 if converged else 1


def _count_seeds_note(args):
    """Best-effort seed-count line for the report (informational only)."""
    parts = []
    if args.candidate_seeds is not None:
        parts.append(f"candidate N={args.candidate_seeds}")
    if args.reference_seeds is not None:
        parts.append(f"reference N={args.reference_seeds}")
    return ("  seed counts      : " + ", ".join(parts)) if parts else None


def main(argv=None):
    ap = argparse.ArgumentParser(
        description="W1.6 z_comb verifier. Default mode 'gate' is the 5-sigma "
                    "pass/fail; mode 'step1-residual' is the cross-pipeline "
                    "systematic-residual MEASUREMENT vs a pre-committed "
                    "relative-deviation ceiling (RED-CLARIFICATION-STEP1).")
    ap.add_argument("--mode", choices=["gate", "step1-residual"], default="gate",
                    help="gate (default): 5-sigma z_comb pass/fail. "
                         "step1-residual: measure the identical-qubit residual "
                         "(sigma_sys) and call convergence vs --max-rel-dev.")
    ap.add_argument("--candidate", required=True,
                    help="engine aggregated_autocorr.dat (device-calibrated arm)")
    ap.add_argument("--reference",
                    default="examples/reference/floquet_dtc_q10_autocorr.csv")
    ap.add_argument("--candidate-seeds", type=int, default=None,
                    help="informational: seeds the candidate was aggregated over")
    ap.add_argument("--reference-seeds", type=int, default=None,
                    help="informational: seeds the reference was aggregated over")
    ap.add_argument("--report", default=None,
                    help="optional CSV path for the per-kick table "
                         "(z table in gate mode; residual table in step1 mode)")
    ap.add_argument("--floor", type=float, default=STEP1_FLOOR,
                    help="step1: |ref| floor for the relative-deviation metric "
                         f"(default {STEP1_FLOOR})")
    ap.add_argument("--max-rel-dev", type=float, default=STEP1_MAX_REL_DEV,
                    help="step1: PRE-COMMITTED convergence ceiling, max per-kick "
                         f"relative deviation (default {STEP1_MAX_REL_DEV} = 2%%)")
    ap.add_argument("--trend-warn", type=float, default=STEP1_TREND_WARN,
                    help="step1: warn if the rel-dev-vs-kick slope exceeds this")
    args = ap.parse_args(argv)

    try:
        candidate = load_aggregated_dat(args.candidate)
        ref_mean, ref_sem = load_reference(args.reference)
    except (OSError, ValueError, KeyError) as e:
        print(f"[z-comb-verify] STRUCTURAL: {e}", file=sys.stderr)
        return 3
    if not candidate:
        print(f"[z-comb-verify] STRUCTURAL: no data rows in {args.candidate}",
              file=sys.stderr)
        return 3

    if args.mode == "step1-residual":
        return _run_step1_residual(args, candidate, ref_mean, ref_sem)

    try:
        res = verify(candidate, ref_mean, ref_sem)
    except ValueError as e:
        print(f"[z-comb-verify] STRUCTURAL: {e}", file=sys.stderr)
        return 3

    if args.report:
        with open(args.report, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["kick", "cand_mean", "cand_sem",
                        "ref_mean", "ref_sem", "z_combined"])
            for k, cm, cs, rm, rs, z in res["rows"]:
                zf = f"{z:.2f}" if math.isfinite(z) else "inf"
                w.writerow([k, f"{cm:.4f}", f"{cs:.4f}",
                            f"{rm:.4f}", f"{rs:.4f}", zf])

    wz = f"{res['worst_z']:.2f}" if math.isfinite(res["worst_z"]) else "inf"
    print("=" * 64)
    print("  W1.6 Z_COMB GATE VERIFIER")
    print("=" * 64)
    print(f"  candidate        : {args.candidate}")
    print(f"  reference        : {args.reference}")
    note = _count_seeds_note(args)
    if note:
        print(note)
    print(f"  kicks compared   : {res['n_compared']}")
    print(f"  worst z_combined : {wz}")
    print(f"  kicks > {SIGMA_FLAG:.0f} sigma  : {res['n_flag']}  (flagged for review)")
    print(f"  kicks > {SIGMA_FAIL:.0f} sigma  : {res['n_fail']}  (gate-failing)")
    if res["n_flag"] and not res["n_fail"]:
        print("  NOTE: kicks beyond 3 sigma — investigate, but within gate "
              "tolerance.", file=sys.stderr)
    print("-" * 64)
    print(f"  VERDICT          : {'PASS' if res['passed'] else 'FAIL'}")
    print("=" * 64)
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
