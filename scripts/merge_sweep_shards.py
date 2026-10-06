#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Post-job merge for the Workstream-B cross-node fan-out (type-dispatched).

After a multi-node sweep where each rank wrote its own ``sweep_rank{r}.h5`` +
``campaign_manifest_rank{r}.json`` (HPCQC_SWEEP_SHARD=1), this CLI:

  1. unions the per-rank HDF5 into ``sweep.h5`` (disjoint per-unit groups ->
     straight copy-in);
  2. selects the REDUCER by the sweep's experiment type (NOT hardcoded —
     RED-RULING-MERGE-CLI-FOLLOWUP §3(a)/(b)): byo_circuit ->
     ``ByoAutocorrReducer`` (aggregates each complete (placement,env) seed series
     via the certified ``aggregate_byo_autocorr`` into ``byo_dat/``);
     characterization | vqe_sweep -> ``BatteryReducer`` (identity reduce, no
     .dat); anything else / a mixed config / no type -> FAIL LOUD;
  3. asserts each EXTRACTED group has its COMPLETE instance series — the actual
     seed_list from --config (or --seeds) — failing loud on a short count (the
     partial-group guard, RED-RULING-WORKSTREAM-B §2), and fails loud on a
     vacuous merge (empty union, or a reducer that matched 0 records — §3(c));
  4. (option (i)) when the engine wrote an expected-group inventory
     (``campaign_expected.json``, on a fresh shard run), asserts the unioned
     GROUP SET equals it — catching a WHOLLY-ABSENT group (a dropped unit / lost
     rank) the per-group short-count guard cannot see. With ``--nranks`` given, a
     whole-rank-FILE drop fails loud at discovery (i-a); a group dropped from a
     present file fails loud at this group-set assert (i-b);
  5. concats the per-rank manifests into ``campaign_manifest.json``.

NOTE (RED-RULING-PATCH43-VERIFY-AND-INVENTORY-DESIGN option (i)): the step-3
completeness assert is a PARTIAL lost-shard guard (a present group missing
instances). The wholly-absent-group case (a group that vanished entirely, the
common whole-rank-loss shape when ``num_placements % nranks == 0``) is closed by
step 4's inventory check. Q1: a MULTI-RANK ``BatteryReducer`` merge with NO
inventory fails loud (never silently reverts to the partial guard); single-rank /
gate merges pass ``expected_groups=None`` and skip it. The BYO inventory
generator is DEBT — until it exists a single-seed-BYO multi-node campaign must
NOT bank results (multi-seed BYO stays clear under the short-count guard);
battery multi-node banks results once the (i-b) gate passes green.

The complete-run result is byte-identical to a single-node run of the same
units; the byte-identity gate (tests/fanout_byte_identity_gate.py) proves it.
Requires h5py + numpy (the container); run on LUMI after the multi-node job, or
via the gate.

Usage:
  python3 scripts/merge_sweep_shards.py --output-dir <dir> --config <sweep.yaml>
  python3 scripts/merge_sweep_shards.py --output-dir <dir> --seeds 0,1 \
      --experiment-type byo_circuit
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)


# Reducers for which the engine writes an expected-group inventory
# (campaign_expected.json). Q1: on a MULTI-RANK merge with one of these, a
# MISSING inventory fails loud rather than silently reverting to the partial
# short-count guard. BatteryReducer has the generator (option (i)); the
# ByoAutocorrReducer generator is DEBT (required before a single-seed BYO
# multi-node campaign banks results) — until it exists, BYO is NOT listed here,
# so a BYO merge skips the group-set check (multi-seed BYO stays clear under the
# existing short-count guard; single-seed BYO multi-node must not bank).
_REDUCERS_WITH_INVENTORY = {"BatteryReducer"}


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


def _read_experiment_types(config_path: str) -> set[str]:
    """The distinct experiment ``type``(s) in the sweep YAML — the same per-
    experiment key the engine reads (default 'characterization', matching
    sweep_engine.py). The reducer is selected from this set; a mixed set is
    rejected by ``select_reducer`` (a single-reducer merge cannot carry a mixed
    sweep)."""
    import yaml

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    types: set[str] = set()
    for exp in cfg.get("sweep", {}).get("experiments", []):
        types.add(exp.get("type", "characterization"))
    if not types:
        raise SystemExit(f"merge: no experiments found in {config_path}")
    return types


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Merge cross-node sweep shards (reducer selected by type)."
    )
    ap.add_argument("--output-dir", required=True,
                    help="dir holding sweep_rank*.h5 + campaign_manifest_rank*.json")
    ap.add_argument("--config", help="sweep YAML to read the expected seed_list "
                                     "and the experiment type from")
    ap.add_argument("--seeds", help="explicit expected seeds, comma-separated "
                                    "(overrides --config)")
    ap.add_argument("--experiment-type",
                    choices=("byo_circuit", "characterization", "vqe_sweep"),
                    default=None,
                    help="reducer-selection override (wins over --config-derived "
                         "type); required if --config is not given")
    ap.add_argument("--hdf5-name", default="sweep.h5",
                    help="name of the merged HDF5 (default: sweep.h5)")
    ap.add_argument("--nranks", type=int, default=None,
                    help="expected number of rank shards (= SLURM_NNODES). When "
                         "given, a sweep_rank*.h5 count != nranks fails loud at "
                         "discovery (a whole-rank-FILE drop) — the cheap "
                         "missing-file guard (option (i-a))")
    args = ap.parse_args(argv)

    from lumi_hpc_qc.sweep.fanout_merge import merge_shards, select_reducer

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

    # ── Option (i-a): whole-rank-FILE drop guard. When --nranks is given (the
    #    production launcher passes SLURM_NNODES), a shard-file count that does
    #    not match fails loud at discovery, before the union runs. ──
    if args.nranks is not None and len(rank_h5s) != args.nranks:
        raise SystemExit(
            f"merge: found {len(rank_h5s)} sweep_rank*.h5 in {out} but "
            f"--nranks={args.nranks} — a whole rank shard file is missing (or "
            f"extra). Refusing to merge an incomplete shard set (option (i-a))."
        )

    manifests = sorted(glob.glob(os.path.join(out, "campaign_manifest_rank*.json")))

    config_types = _read_experiment_types(args.config) if args.config else None
    try:
        reducer = select_reducer(config_types, args.experiment_type, seeds)
    except ValueError as e:
        raise SystemExit(f"merge: {e}")

    # ── Option (i)/(i-b): the expected-group inventory. The engine writes
    #    campaign_expected.json on a fresh shard run; the merge asserts the
    #    unioned group set equals it, catching a WHOLLY-ABSENT group (a dropped
    #    unit / lost rank the per-group short-count guard cannot see). Q1: on a
    #    MULTI-RANK merge with an inventory-bearing reducer, a MISSING inventory
    #    fails loud — never silently revert to the partial guard. Single-rank /
    #    no-generator merges pass expected_groups=None (skip), preserving the
    #    prior behavior and keeping the byte-identity gate green. ──
    expected_groups = None
    inv_path = os.path.join(out, "campaign_expected.json")
    if os.path.exists(inv_path):
        from lumi_hpc_qc.sweep.battery_paths import inventory_from_json
        try:
            with open(inv_path, encoding="utf-8") as f:
                expected_groups = inventory_from_json(json.load(f))
        except ValueError as e:
            raise SystemExit(f"merge: {e}")
    elif len(rank_h5s) > 1 and type(reducer).__name__ in _REDUCERS_WITH_INVENTORY:
        raise SystemExit(
            f"merge: multi-rank ({len(rank_h5s)} shards) "
            f"{type(reducer).__name__} merge but no campaign_expected.json in "
            f"{out} — the wholly-absent-group guard would be silently disabled. "
            f"A fresh shard run writes the inventory; re-run to regenerate it, "
            f"or merge a single rank. Refusing to merge without it (Q1)."
        )

    reduced = merge_shards(
        rank_h5_paths=rank_h5s,
        out_h5_path=os.path.join(out, args.hdf5_name),
        reducer=reducer,
        out_root=os.path.join(out, "byo_dat"),
        rank_manifest_paths=manifests or None,
        out_manifest_path=(
            os.path.join(out, "campaign_manifest.json") if manifests else None
        ),
        expected_groups=expected_groups,
    )
    _inv_note = (
        f"group-set checked vs {len(expected_groups)} expected"
        if expected_groups is not None
        else "group-set check skipped (no inventory)"
    )
    print(f"merge: unioned {len(rank_h5s)} rank shard(s) -> "
          f"{os.path.join(out, args.hdf5_name)}; "
          f"reducer={type(reducer).__name__}; reduced {len(reduced)} group(s); "
          f"expected seeds={seeds}; {_inv_note}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
