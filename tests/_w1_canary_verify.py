#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""W1 canary byte-match verifier.

Not a pytest test — the leading underscore in the filename keeps pytest from
trying to collect it. Invoked by tests/slurm_w1_canary.sh as a regular Python
script.

Walks a workdir for engine-produced ``instance_NN_autocorr.dat`` files (the
output of ``aggregate_byo_autocorr`` with ``write_per_instance=True``), groups
them by arm (``noiseless`` / ``device_calibrated``), computes SHA-256, and
compares against the in-tree oracle at
``evidence/W1/gate2_canary/sha256_oracle.txt``.

Acceptance bar (RED-RESP-W1-PARALLELISM-AND-OOM-ROOTCAUSE-v1.4 §6 / F3,
hardened per RED-RESP-W1.3-VERIFY-AND-W1.4-CAP-RULINGS NF3):
PASS iff (1) the ``device_calibrated`` arm matches the oracle SHAs for BOTH
seeds, AND (2) the ``noiseless`` arm is present and does NOT match the
device-cal oracle. The oracle pins the device_calibrated arm of the 2-seed
corpus (banked at session V19). The old "iff at least one arm matches" bar
was unsound: it keyed only on a path segment, so an arm-label swap (device-cal
physics written under ``noiseless/``) would have PASSED while mislabeling
arms. Requiring the device-cal arm specifically AND noiseless distinctness
(a noiseless arm matching a device-cal oracle means noise was not applied)
closes that hole before this becomes the W1.6 standing gate. The full 40-seed
gate-2 reproduction (per-kick z_comb vs the runner reference) is separate, at
W1.6 — this verifier is the same-path 2-seed byte-match regression.

Exit code: 0 on PASS, 1 on FAIL.

Usage:
  python3 tests/_w1_canary_verify.py --workdir <path> --oracle <path>
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import os
import re
import sys


def _parse_oracle(oracle_path: str) -> dict[int, str]:
    """Parse a sha256sum-formatted oracle file into ``{seed_idx: sha256_hex}``.

    Recognises lines of the shape ``<sha>  <path>`` where ``<path>`` contains
    a ``seed_NN_instance`` segment (the W1 canary banking convention). Other
    lines, blank lines, and ``#`` comments are skipped silently.
    """
    seed_to_sha: dict[int, str] = {}
    with open(oracle_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 1)
            if len(parts) != 2:
                continue
            sha, path = parts
            base = os.path.basename(path)
            m = re.search(r"seed_(\d+)_instance", base)
            if m:
                seed_to_sha[int(m.group(1))] = sha.lower()
    return seed_to_sha


def _find_engine_outputs(workdir: str) -> dict[str, dict[int, str]]:
    """Walk ``workdir`` for ``instance_NN_autocorr.dat`` files; group by arm.

    The engine writes
    ``<workdir>/sweep_output/<sweep_id>/byo_dat/<script>/<phys>/<arm>/instance_NN_autocorr.dat``
    (per ``_execute_byo_group`` post-W1.3 + ``aggregate_byo_autocorr``).
    We recognise the arm by looking for ``noiseless`` / ``device_calibrated``
    anywhere in the path segments; this is robust to engine path-format
    revisions that don't touch the arm-segment convention.

    Returns ``{arm: {seed_idx: absolute_path}}``.
    """
    by_arm: dict[str, dict[int, str]] = {}
    candidates = glob.glob(
        os.path.join(workdir, "**", "instance_*_autocorr.dat"),
        recursive=True,
    )
    for path in candidates:
        parts = path.split(os.sep)
        arm = next(
            (p for p in parts if p in ("noiseless", "device_calibrated")),
            None,
        )
        if arm is None:
            continue
        m = re.search(r"instance_(\d+)_autocorr\.dat", os.path.basename(path))
        if not m:
            continue
        seed_idx = int(m.group(1))
        by_arm.setdefault(arm, {})[seed_idx] = path
    return by_arm


def _sha256_of(path: str) -> str:
    """SHA-256 of a file's bytes, returned as lowercase hex."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="W1 canary byte-match verifier.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--workdir",
        required=True,
        help="Engine output workdir (contains sweep_output/<sweep_id>/...)",
    )
    parser.add_argument(
        "--oracle",
        required=True,
        help="Path to evidence/W1/gate2_canary/sha256_oracle.txt",
    )
    args = parser.parse_args()

    print("=== W1 Canary byte-match verifier ===")
    print(f"Workdir: {args.workdir}")
    print(f"Oracle:  {args.oracle}")

    if not os.path.isfile(args.oracle):
        print(f"\nFAIL: oracle file not found: {args.oracle}")
        return 1
    if not os.path.isdir(args.workdir):
        print(f"\nFAIL: workdir not found: {args.workdir}")
        return 1

    oracle_sha = _parse_oracle(args.oracle)
    print(f"\nOracle SHAs ({len(oracle_sha)} seed(s)):")
    for sidx in sorted(oracle_sha):
        print(f"  seed {sidx:02d}: {oracle_sha[sidx]}")

    by_arm = _find_engine_outputs(args.workdir)
    print(f"\nFound arms in engine output: {sorted(by_arm.keys())}")
    for arm, seeds in sorted(by_arm.items()):
        print(f"  {arm}: {len(seeds)} per-instance dat(s) "
              f"(seeds {sorted(seeds.keys())})")

    if not by_arm:
        print("\nW1 CANARY ACCEPTANCE: FAILED — "
              "no instance_NN_autocorr.dat files found under workdir.")
        return 1

    def _arm_matches_oracle(arm: str) -> tuple[bool, bool]:
        """Return (present, matches_all_seeds) for one arm vs the oracle.

        ``present`` is True if the arm has at least one .dat for an oracle seed.
        ``matches_all_seeds`` is True only if every oracle seed is present AND
        byte-matches. Per-seed lines are printed for the audit trail.
        """
        seeds = by_arm.get(arm, {})
        print(f"\n--- arm: {arm} ---")
        present = False
        all_match = True
        for seed_idx in sorted(oracle_sha):
            engine_path = seeds.get(seed_idx)
            if engine_path is None:
                print(f"  seed {seed_idx:02d}: MISSING engine output for this arm")
                all_match = False
                continue
            present = True
            engine_sha = _sha256_of(engine_path)
            want_sha = oracle_sha[seed_idx]
            ok = engine_sha == want_sha
            mark = "OK" if ok else "MISMATCH"
            print(f"  seed {seed_idx:02d}: "
                  f"engine={engine_sha[:16]}.. oracle={want_sha[:16]}.. {mark}")
            if not ok:
                all_match = False
        return present, (present and all_match)

    # NF3: the oracle pins the device_calibrated arm. Two specific assertions
    # replace the old "iff any arm matches" (which an arm-label swap could fool):
    #   (1) device_calibrated MUST match both seeds; and
    #   (2) noiseless MUST be present and MUST NOT match the device-cal oracle
    #       (a noiseless arm matching it means noise was not applied).
    dc_present, dc_matches = _arm_matches_oracle("device_calibrated")
    nl_present, nl_matches = _arm_matches_oracle("noiseless")

    failures: list[str] = []
    if not dc_present:
        failures.append("device_calibrated arm is MISSING (expected the heavy "
                        "arm the oracle pins).")
    elif not dc_matches:
        failures.append("device_calibrated arm does NOT byte-match the oracle "
                        "for both seeds (the same-path reproduction regressed).")
    if not nl_present:
        failures.append("noiseless arm is MISSING (expected both arms to run; "
                        "its presence + distinctness proves arm labelling).")
    elif nl_matches:
        failures.append("noiseless arm MATCHES the device-cal oracle — noise "
                        "was not applied, or arms are mislabelled (NF3).")

    if failures:
        print("\nW1 CANARY ACCEPTANCE: FAILED —")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nW1 CANARY ACCEPTANCE: ALL CHECKS PASSED")
    print("  device_calibrated: byte-match on both seeds; "
          "noiseless: present and distinct (noise applied).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
