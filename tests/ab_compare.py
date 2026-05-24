#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""A/B compare two Floquet runs' per-instance autocorrelators.

    python3 ab_compare.py <run_dir_A> <run_dir_B> [--exact] [--atol X] [--sigmas K]

Loads instance_NN_full.json from each directory, matches instances by
instance_id, and checks the per-kick autocorrelator arrays agree. Intended to
confirm the floquet_runner -> floquet_runner_v2 (prepare() seam) refactor is
behaviour-preserving in one command instead of eyeballing diffs.

Two modes:

  statistical (default)
      The simulators are NOT seeded (the runner passes no seed_simulator), so
      shot sampling differs run-to-run even for an identical setup. Each
      autocorrelator carries a shot-noise standard error ~ 1/sqrt(shots), so
      the difference between two independent runs has sigma ~ sqrt(2/shots) per
      kick. PASS if max|delta| over all kicks <= sigmas * sigma (default
      sigmas=6). This confirms the refactor changed nothing beyond sampling
      noise. With shots=0 (exact statevector / density-matrix eval) sigma=0 and
      this collapses to an exact check automatically.

  exact (--exact)
      PASS only if max|delta| <= atol (default 1e-9). Use this when BOTH runs
      were produced with a fixed simulator seed, which makes the autocorrelators
      bit-reproducible and turns the A/B into strict equality. (The runner does
      not seed by default; see the note printed at the end if you need this.)

Note: floquet_runner_v2 adds `circuit_metrics` / `fake_backend` keys that v1's
JSON lacks. This tool compares ONLY `autocorrelators`, so that schema
difference is irrelevant and expected -- it is not a regression.

Exit code 0 = all matched instances pass; 1 = any fail or structural mismatch.
"""
import os
import sys
import json
import glob
import math
import argparse


def load_run(directory):
    """Return {instance_id: parsed_json} for one run directory."""
    runs = {}
    pattern = os.path.join(directory, "instance_*_full.json")
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"  WARN: could not read {path}: {e}")
            continue
        iid = data.get("instance_id")
        if iid is None:
            # Fall back to the integer in the filename.
            base = os.path.basename(path)
            digits = "".join(c for c in base if c.isdigit())
            iid = int(digits) if digits else base
        runs[iid] = data
    return runs


def _max_rms(a, b):
    """(max_abs_diff, rms_diff, has_nan) for two equal-length sequences."""
    has_nan = False
    sq = 0.0
    mx = 0.0
    for x, y in zip(a, b):
        if x is None or y is None or math.isnan(x) or math.isnan(y):
            has_nan = True
            continue
        d = abs(x - y)
        mx = max(mx, d)
        sq += d * d
    n = len(a)
    rms = math.sqrt(sq / n) if n else 0.0
    return mx, rms, has_nan


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("dir_a", help="first run directory (e.g. v1 / reference)")
    ap.add_argument("dir_b", help="second run directory (e.g. v2 / seam)")
    ap.add_argument("--exact", action="store_true",
                    help="strict equality (use only for seeded runs)")
    ap.add_argument("--atol", type=float, default=1e-9,
                    help="absolute tolerance for --exact mode (default 1e-9)")
    ap.add_argument("--sigmas", type=float, default=6.0,
                    help="shot-noise multiplier for statistical mode (default 6)")
    args = ap.parse_args()

    runs_a = load_run(args.dir_a)
    runs_b = load_run(args.dir_b)

    if not runs_a or not runs_b:
        print(f"ERROR: no instance_*_full.json found "
              f"({'A empty' if not runs_a else ''}"
              f"{' and ' if not runs_a and not runs_b else ''}"
              f"{'B empty' if not runs_b else ''}).")
        return 1

    ids_a, ids_b = set(runs_a), set(runs_b)
    common = sorted(ids_a & ids_b, key=lambda x: (isinstance(x, str), x))
    only_a, only_b = ids_a - ids_b, ids_b - ids_a

    print(f"ab_compare:")
    print(f"  A = {args.dir_a}  ({len(ids_a)} instances)")
    print(f"  B = {args.dir_b}  ({len(ids_b)} instances)")
    if only_a:
        print(f"  WARN: only in A: {sorted(only_a)}")
    if only_b:
        print(f"  WARN: only in B: {sorted(only_b)}")
    if not common:
        print("ERROR: no instance_id present in both runs.")
        return 1

    mode = "exact" if args.exact else "statistical"
    print(f"  mode: {mode}"
          + (f" (atol={args.atol:g})" if args.exact
             else f" (sigmas={args.sigmas:g})"))
    print("-" * 64)

    failures = []
    structural = bool(only_a or only_b)

    for iid in common:
        da, db = runs_a[iid], runs_b[iid]
        ac_a = da.get("autocorrelators")
        ac_b = db.get("autocorrelators")

        if not isinstance(ac_a, list) or not isinstance(ac_b, list):
            print(f"  instance {iid}: MISSING autocorrelators  FAIL")
            failures.append(iid)
            continue
        if len(ac_a) != len(ac_b):
            print(f"  instance {iid}: kick-count mismatch "
                  f"{len(ac_a)} vs {len(ac_b)}  FAIL")
            failures.append(iid)
            structural = True
            continue

        shots_a = da.get("num_shots", 0) or 0
        shots_b = db.get("num_shots", 0) or 0
        if shots_a != shots_b:
            print(f"  instance {iid}: WARN num_shots differ "
                  f"({shots_a} vs {shots_b}); using {min(shots_a, shots_b)} "
                  f"for the shot-noise estimate")
        shots = min(shots_a, shots_b)

        mx, rms, has_nan = _max_rms(ac_a, ac_b)

        if args.exact:
            tol = args.atol
        else:
            sigma = math.sqrt(2.0 / shots) if shots > 0 else 0.0
            tol = args.sigmas * sigma

        ok = (mx <= tol + 1e-12) and not has_nan
        nan_note = "  [NaN present]" if has_nan else ""
        status = "PASS" if ok else "FAIL"
        print(f"  instance {iid}: kicks={len(ac_a):3d}  "
              f"max|Δ|={mx:.4f}  rms={rms:.4f}  tol={tol:.4f}  "
              f"{status}{nan_note}")
        if not ok:
            failures.append(iid)

    print("-" * 64)
    n_ok = len(common) - len(failures)
    passed = (not failures) and (not structural)

    if not args.exact:
        # Make the statistical caveat explicit so a PASS isn't over-read.
        print("note: statistical mode tolerates shot-sampling differences. For a "
              "strict\n      bit-for-bit check, run both with a fixed simulator "
              "seed and pass --exact.")

    if passed:
        print(f"RESULT: PASS ({n_ok}/{len(common)} matched instances within "
              f"tolerance)")
        return 0
    print(f"RESULT: FAIL ({len(failures)} of {len(common)} instances over "
          f"tolerance{'; instance-set mismatch' if structural else ''})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
