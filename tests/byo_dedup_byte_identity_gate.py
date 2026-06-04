"""§5.4 noiseless-dedup byte-identity GATE.

RED-RULING-BYO-FLAT-DISPATCH-AND-NOISELESS-DEDUP §5.4. Runs the multi-observable
fixture single-node TWICE — byo_noiseless_dedup false (reference) vs true — and
asserts the full output is identical:

  * the byo_dat .dat tree byte-for-byte (the gate-2 comparison artifact), and
  * the /byo sweep.h5 subtree at the DATASET level (path-keyed, non-physics attrs
    excluded). Dataset-level — NOT raw .h5 bytes — is deliberate: the dedup
    broadcast writes the per-placement noiseless records grouped-by-source rather
    than interleaved, so HDF5 group CREATION ORDER differs while content does not
    (the disclosed Patch-C caveat). Comparing by dataset path is order-independent.

It also asserts dedup ENGAGED (strictly fewer units dispatched with it on), so a
byte-identity pass cannot be a no-op (e.g. an unwired flag building the same units).

Reuses the certified comparators from the fan-out byte-identity gate. Run inside
the LUMI qiskit container (the sweeps need aer).

Usage:
  python3 tests/byo_dedup_byte_identity_gate.py \
      --config examples/byo/floquet_dtc_q10_multiobs_gate.yaml --workdir <dir>
"""
import argparse
import copy
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fanout_byte_identity_gate import (  # noqa: E402  (certified comparators)
    _spawn_run, _assert_dats_identical, _assert_h5_subtree_equal,
)


def _write_cfg(base, output_dir, dedup, dest):
    cfg = copy.deepcopy(base)
    cfg["sweep"]["output_dir"] = output_dir
    for exp in cfg["sweep"]["experiments"]:
        exp["byo_noiseless_dedup"] = bool(dedup)
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


def _dispatched(stdout):
    return sum(int(m) for m in re.findall(r"dispatching (\d+) unit", stdout))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",
                    default="examples/byo/floquet_dtc_q10_multiobs_gate.yaml")
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args(argv)

    base = yaml.safe_load(open(args.config, encoding="utf-8"))
    os.makedirs(args.workdir, exist_ok=True)
    off_dir = os.path.join(args.workdir, "dedup_off")
    on_dir = os.path.join(args.workdir, "dedup_on")
    os.makedirs(off_dir, exist_ok=True)
    os.makedirs(on_dir, exist_ok=True)
    off_cfg = os.path.join(args.workdir, "cfg_dedup_off.yaml")
    on_cfg = os.path.join(args.workdir, "cfg_dedup_on.yaml")
    _write_cfg(base, off_dir, False, off_cfg)
    _write_cfg(base, on_dir, True, on_cfg)

    print("═══ dedup OFF (reference) ═══")
    p_off = _spawn_run(off_cfg, shard=False)
    print("═══ dedup ON ═══")
    p_on = _spawn_run(on_cfg, shard=False)

    fails = []
    _assert_dats_identical(off_dir, on_dir, "dedup", fails)
    _assert_h5_subtree_equal(
        os.path.join(off_dir, "sweep.h5"),
        os.path.join(on_dir, "sweep.h5"),
        "byo", "dedup", fails,
    )

    d_off, d_on = _dispatched(p_off.stdout), _dispatched(p_on.stdout)
    print(f"[dedup] units dispatched: off={d_off} on={d_on} "
          f"(on<off proves dedup engaged; equal output proves it is safe)")
    if not (d_off > d_on > 0):
        fails.append(
            f"[dedup] dedup did NOT engage (dispatched off={d_off} on={d_on}); "
            f"a byte-identity pass here would be a no-op, not a proof"
        )

    if fails:
        print("\n═══ §5.4 DEDUP GATE FAILED ═══")
        for f in fails:
            print("  " + f)
        return 1
    print("\n═══ §5.4 DEDUP GATE PASSED ═══")
    return 0


if __name__ == "__main__":
    sys.exit(main())
