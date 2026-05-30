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
        description="W1.6 z_comb gate verifier (pass/fail).")
    ap.add_argument("--candidate", required=True,
                    help="engine aggregated_autocorr.dat (device-calibrated arm)")
    ap.add_argument("--reference",
                    default="examples/reference/floquet_dtc_q10_autocorr.csv")
    ap.add_argument("--candidate-seeds", type=int, default=None,
                    help="informational: seeds the candidate was aggregated over")
    ap.add_argument("--reference-seeds", type=int, default=None,
                    help="informational: seeds the reference was aggregated over")
    ap.add_argument("--report", default=None,
                    help="optional CSV path for the per-kick z table")
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
