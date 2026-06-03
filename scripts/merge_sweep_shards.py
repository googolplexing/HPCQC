#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Post-job merge for the Workstream-B cross-node fan-out (BYO path).

After a multi-node sweep where each rank wrote its own ``sweep_rank{r}.h5`` +
``campaign_manifest_rank{r}.json`` (HPCQC_SWEEP_SHARD=1), this CLI:

  1. unions the per-rank HDF5 into ``sweep.h5`` (disjoint per-unit groups ->
     straight copy-in);
  2. asserts each (placement, env) group has the COMPLETE seed series — the
     actual seed_list from --config (or --seeds) — failing loud on a gap (the
     short-count guard, RED-RULING-WORKSTREAM-B §2);
  3. aggregates each complete group via the certified ``aggregate_byo_autocorr``
     into ``byo_dat/`` (byte-format-identical to the single-node .dat tree);
  4. concats the per-rank manifests into ``campaign_manifest.json``.

The result is byte-identical to a single-node run of the same units; the
byte-identity gate (tests/fanout_byte_identity_gate.py) proves it. Requires h5py
+ numpy (the container); run on LUMI after the multi-node job, or via the gate.

Usage:
  python3 scripts/merge_sweep_shards.py --output-dir <dir> --config <sweep.yaml>
  python3 scripts/merge_sweep_shards.py --output-dir <dir> --seeds 0,1
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


def _read_seed_list(config_path: str) -> list[int]:
    """The run's seed_list from the sweep YAML — the same config the run used, so
    'expected_instances' is wired to the actual seeds. Assumes a uniform
    seed_list across experiments (true for the BYO echo campaign and the gate);
    a per-experiment expected set is a future refinement if they ever differ."""
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    seeds: set[int] = set()
    for exp in cfg.get("sweep", {}).get("experiments", []):
        for s in exp.get("seed_list", []):
            seeds.add(int(s))
    if not seeds:
        raise SystemExit(f"merge: no seed_list found in {config_path}")
    return sorted(seeds)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Merge cross-node sweep shards (BYO).")
    ap.add_argument("--output-dir", required=True,
                    help="dir holding sweep_rank*.h5 + campaign_manifest_rank*.json")
    ap.add_argument("--config", help="sweep YAML to read the expected seed_list from")
    ap.add_argument("--seeds", help="explicit expected seeds, comma-separated "
                                    "(overrides --config)")
    ap.add_argument("--hdf5-name", default="sweep.h5",
                    help="name of the merged HDF5 (default: sweep.h5)")
    args = ap.parse_args(argv)

    from lumi_hpc_qc.sweep.fanout_merge import ByoAutocorrReducer, merge_shards

    if args.seeds:
        seeds = [int(x) for x in args.seeds.split(",") if x.strip() != ""]
    elif args.config:
        seeds = _read_seed_list(args.config)
    else:
        raise SystemExit("merge: provide --config or --seeds for the expected seed_list")

    out = args.output_dir
    rank_h5s = sorted(glob.glob(os.path.join(out, "sweep_rank*.h5")))
    if not rank_h5s:
        raise SystemExit(f"merge: no sweep_rank*.h5 in {out}")
    manifests = sorted(glob.glob(os.path.join(out, "campaign_manifest_rank*.json")))

    reducer = ByoAutocorrReducer(expected_seeds=seeds)  # default byo_dat layout
    reduced = merge_shards(
        rank_h5_paths=rank_h5s,
        out_h5_path=os.path.join(out, args.hdf5_name),
        reducer=reducer,
        out_root=os.path.join(out, "byo_dat"),
        rank_manifest_paths=manifests or None,
        out_manifest_path=(
            os.path.join(out, "campaign_manifest.json") if manifests else None
        ),
    )
    print(f"merge: unioned {len(rank_h5s)} rank shard(s) -> "
          f"{os.path.join(out, args.hdf5_name)}; "
          f"aggregated {len(reduced)} (placement,env,observable) group(s); "
          f"expected seeds={seeds}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
