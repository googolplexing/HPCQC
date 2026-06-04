# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""RED-DIRECTIVE-PROBE-SKIP-WHEN-NON-BINDING §6 — A/B byte-identity GATE.

Runs the multi-observable fixture single-node TWICE on IDENTICAL config — once
with the probe-skip ACTIVE (default; at 10q memory is non-binding so the engine
skips the D1/D2 probe) and once with ``HPCQC_FORCE_PROBE=1`` (forces the probe)
— and asserts:

  * the byo_dat .dat tree byte-for-byte, and
  * the /byo sweep.h5 subtree at the DATASET level (path-keyed, non-physics
    attrs excluded) — the same certified comparators the fan-out and dedup gates
    use.

The cap governs only pool CONCURRENCY, never the per-unit computation (each unit
is deterministic via ``resolve_instance_seed``), so skip-vs-probe MUST be
byte-identical — this gate proves the optimization is physics-invariant.

NON-VACUITY: it also asserts the two arms actually took DIFFERENT cap paths — the
skip arm's footer carries ``[skip:mem_non_binding]`` and the forced arm's carries
``[probe:device_calibrated_VmHWM]`` — so a byte-identity pass cannot be a no-op
(e.g. an unwired toggle that skipped on both arms, which would pass trivially).

Run inside the LUMI qiskit container (the sweeps need aer).

Usage:
  python3 tests/byo_probe_skip_byte_identity_gate.py \
      --config examples/byo/floquet_dtc_q10_multiobs_gate.yaml --workdir <dir>
"""
import argparse
import copy
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fanout_byte_identity_gate import (  # noqa: E402  (certified comparators)
    _spawn_run, _assert_dats_identical, _assert_h5_subtree_equal,
)

_SKIP_TAG = "skip:mem_non_binding"
_PROBE_TAG = "probe:device_calibrated_VmHWM"


def _write_cfg(base, output_dir, dest):
    cfg = copy.deepcopy(base)
    cfg["sweep"]["output_dir"] = output_dir
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="examples/byo/floquet_dtc_q10_multiobs_gate.yaml")
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args(argv)

    base = yaml.safe_load(open(args.config, encoding="utf-8"))
    os.makedirs(args.workdir, exist_ok=True)
    skip_dir = os.path.join(args.workdir, "skip")
    probe_dir = os.path.join(args.workdir, "probe")
    os.makedirs(skip_dir, exist_ok=True)
    os.makedirs(probe_dir, exist_ok=True)
    skip_cfg = os.path.join(args.workdir, "cfg_skip.yaml")
    probe_cfg = os.path.join(args.workdir, "cfg_probe.yaml")
    _write_cfg(base, skip_dir, skip_cfg)
    _write_cfg(base, probe_dir, probe_cfg)

    # _spawn_run inherits the parent os.environ, so toggling it here selects the
    # cap path for each arm on otherwise-identical config.
    print("═══ probe-skip ACTIVE (default path) ═══")
    os.environ.pop("HPCQC_FORCE_PROBE", None)
    p_skip = _spawn_run(skip_cfg, shard=False)
    print("═══ probe FORCED (HPCQC_FORCE_PROBE=1) ═══")
    os.environ["HPCQC_FORCE_PROBE"] = "1"
    try:
        p_probe = _spawn_run(probe_cfg, shard=False)
    finally:
        os.environ.pop("HPCQC_FORCE_PROBE", None)

    fails = []
    _assert_dats_identical(skip_dir, probe_dir, "probe-skip", fails)
    _assert_h5_subtree_equal(
        os.path.join(skip_dir, "sweep.h5"),
        os.path.join(probe_dir, "sweep.h5"),
        "byo", "probe-skip", fails,
    )

    # NON-VACUITY: prove the arms took DIFFERENT cap paths (else byte-identity is
    # a trivial no-op — e.g. an unwired toggle skipping on both arms).
    skip_skipped = _SKIP_TAG in p_skip.stdout
    skip_probed = _PROBE_TAG in p_skip.stdout
    probe_probed = _PROBE_TAG in p_probe.stdout
    probe_skipped = _SKIP_TAG in p_probe.stdout
    print(
        f"[probe-skip] cap paths: skip-arm(skipped={skip_skipped} "
        f"probed={skip_probed}) probe-arm(probed={probe_probed} "
        f"skipped={probe_skipped}) — skip-arm must skip, probe-arm must probe"
    )
    if not (skip_skipped and not skip_probed):
        fails.append(
            f"[probe-skip] skip arm did NOT skip (no [{_SKIP_TAG}] in footer) — "
            f"the non-binding skip path was not exercised; gate is vacuous"
        )
    if not (probe_probed and not probe_skipped):
        fails.append(
            f"[probe-skip] forced arm did NOT probe (no [{_PROBE_TAG}] in "
            f"footer) — HPCQC_FORCE_PROBE is unwired; gate is vacuous"
        )

    if fails:
        print("\n═══ §6 PROBE-SKIP A/B GATE FAILED ═══")
        for f in fails:
            print("  " + f)
        return 1
    print("\n═══ §6 PROBE-SKIP A/B GATE PASSED ═══")
    print("  skip vs forced-probe: byte-identical .dat + /byo output; the two "
          "arms took distinct cap paths (skip vs probe).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
