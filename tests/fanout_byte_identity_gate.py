#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Byte-identity GATE for the Workstream-B cross-node fan-out (BYO + battery).

Runs a fixture three ways and proves the shard+merge is byte-identical to a
single-node run (RED-RULING-WORKSTREAM-B §6.1/§8.4;
RED-REVIEW-WORKSTREAM-B-INCREMENTS-1-2 §7.3; RED-RULING-ITEM3 for the battery
path). One harness, two paths via --path:

  * --path byo (default): the floquet counts->autocorrelator path. Aggregates
    into byo_dat/ (.dat) and keys the .h5 under /byo. Each (placement, seed, env)
    is a SEPARATE unit, so the shard is env-stratified and the 2-rank run also
    checks §3.3 env-co-residency.
  * --path battery: the synthetic-channel hamiltonian twin battery. NO
    aggregation (identity reduce — RED-RULING-ITEM3 §1); per-unit twin records
    under /devices. Each (seed, placement) unit runs ALL envs inside one worker,
    so co-residency is structural (no stratification, no co-residency check) and
    there is no .dat to compare.

Each path runs three ways: single-node (HPCQC_SWEEP_SHARD unset — the baseline);
2-rank shard (a divisor of the unit count); and 3-rank shard (the NON-dividing
case where a shard off-by-one would hide). Each rank is a separate engine process
with HPCQC_SWEEP_SHARD=1 + SLURM_NNODES/SLURM_NODEID set, so the gate needs only
ONE node (byte-identity is allocation-shape-independent — a unit's output is a
pure function of its inputs). After each shard run the gate merges the rank
shards and asserts the merged sweep.h5 subtree (/byo or /devices) matches
single-node at the DATASET level — NOT raw .h5 bytes (HDF5 container layout
legitimately differs between one writer and a union): identical group/dataset
path-set, exact np.array_equal on every dataset, and equal per-node attributes.
Run-level metadata attrs (experiment_id's per-process sweep_id, wall_time_seconds,
created_at) are not compared — they are not physics (see _NONPHYSICS_ATTRS). The
BYO path additionally asserts the byo_dat .dat tree is byte-
identical (raw bytes — the certified aggregation artifact).

Exit 0 on pass, 1 on any mismatch. Requires the container (qiskit-aer + h5py).

Usage (inside the container, from the repo root):
  python3 tests/fanout_byte_identity_gate.py                 # BYO path (default)
  python3 tests/fanout_byte_identity_gate.py --path battery  # battery path
  python3 tests/fanout_byte_identity_gate.py --config <fixture.yaml> --workdir <dir>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

# Per-path fixtures, HDF5 subtree roots, and (BYO-only) env-co-residency sets.
# The BYO path aggregates counts->autocorrelator into byo_dat/ and keys the .h5
# under /byo; the battery path has no aggregation (identity reduce) and keys the
# per-unit twin records under /devices (RED-RULING-ITEM3).
_FIXTURES = {
    "byo": "examples/byo/floquet_dtc_q10_fanout_gate_8unit.yaml",
    "battery": "examples/battery/tfim_4q_fanout_gate.yaml",
}
_H5_ROOT = {"byo": "byo", "battery": "devices"}
_ENVS = {"noiseless", "device_calibrated"}   # BYO co-residency check set

# Run-level metadata attrs that legitimately vary between a single-node run and
# per-rank shard runs, so they are NOT part of byte-identity — excluded for every
# path (the generalization of the BYO gate's created_at exclusion: "not physics").
# These are provably non-physics:
#   * wall_time_seconds — execution timing; differs even between two identical
#     re-runs of the same unit.
#   * experiment_id — "{sweep_id}_{task}_{placement}_{env}"; differs ONLY in the
#     per-engine-process sweep_id prefix (each rank is its own engine process, so
#     it mints its own sweep_id — same architectural property as the BYO path).
#     The task/placement/env parts are deterministic and ARE verified, via the
#     group path + the seed/noise_config attrs.
#   * created_at — run timestamp (the BYO gate's original exclusion).
# Every PHYSICS attr (best_energy, noise_fingerprint, per_edge_cz_fidelity,
# per_qubit_calibration, placement_score, exact_ground_energy, topology_hash,
# seed, noise_config, calibration_id, ...) and every dataset are still compared.
_NONPHYSICS_ATTRS = frozenset({"experiment_id", "wall_time_seconds", "created_at"})


# ── config helpers ──────────────────────────────────────────────────────────

def _load_cfg(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _write_cfg(base, output_dir, dest):
    import copy
    import yaml
    cfg = copy.deepcopy(base)
    cfg["sweep"]["output_dir"] = output_dir
    with open(dest, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)


# ── engine run (isolated subprocess per rank) ───────────────────────────────

def _spawn_run(cfg_path, *, shard, nranks=1, rank=0):
    env = {k: v for k, v in os.environ.items() if k != "HPCQC_SWEEP_SHARD"}
    if shard:
        env["HPCQC_SWEEP_SHARD"] = "1"
        env["SLURM_NNODES"] = str(nranks)
        env["SLURM_NODEID"] = str(rank)
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--run-one",
         "--config", cfg_path],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(
            f"gate: engine run FAILED (shard={shard} rank={rank}/{nranks}) "
            f"for {cfg_path}"
        )
    return proc


def _merge(output_dir, seeds, path):
    import glob
    from lumi_hpc_qc.sweep.fanout_merge import (
        BatteryReducer, ByoAutocorrReducer, merge_shards,
    )
    rank_h5s = sorted(glob.glob(os.path.join(output_dir, "sweep_rank*.h5")))
    manifests = sorted(
        glob.glob(os.path.join(output_dir, "campaign_manifest_rank*.json"))
    )
    if path == "battery":
        # identity reduce: union + completeness only, no .dat (out_root unused)
        reducer = BatteryReducer(expected_seeds=seeds)
    else:
        reducer = ByoAutocorrReducer(expected_seeds=seeds)  # default byo_dat layout
    merge_shards(
        rank_h5_paths=rank_h5s,
        out_h5_path=os.path.join(output_dir, "sweep.h5"),
        reducer=reducer,
        out_root=os.path.join(output_dir, "byo_dat"),
        rank_manifest_paths=manifests or None,
        out_manifest_path=(
            os.path.join(output_dir, "campaign_manifest.json")
            if manifests else None
        ),
    )


# ── comparators ─────────────────────────────────────────────────────────────

def _collect_dats(byo_dat_dir):
    files = {}
    if not os.path.isdir(byo_dat_dir):
        return files
    for root, _dirs, names in os.walk(byo_dat_dir):
        for fn in names:
            full = os.path.join(root, fn)
            with open(full, "rb") as fh:
                files[os.path.relpath(full, byo_dat_dir)] = fh.read()
    return files


def _norm_attr(v):
    import numpy as np
    if isinstance(v, bytes):
        return v.decode("utf-8", "replace")
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return tuple(v.tolist())
    return v


def _collect_h5_subtree(h5path, root):
    import h5py
    import numpy as np
    ds_vals, node_attrs = {}, {}
    with h5py.File(h5path, "r") as f:
        sub = f.get(root)
        if sub is None:
            return ds_vals, node_attrs

        def visit(name, obj):
            node_attrs[name] = {
                k: _norm_attr(v) for k, v in obj.attrs.items()
                if k not in _NONPHYSICS_ATTRS
            }
            if isinstance(obj, h5py.Dataset):
                ds_vals[name] = np.asarray(obj[()])
        sub.visititems(visit)
    return ds_vals, node_attrs


def _envs_in_rank(h5path):
    import h5py
    envs = set()
    with h5py.File(h5path, "r") as f:
        root = f.get("byo")
        if root is None:
            return envs

        def visit(name, obj):
            if isinstance(obj, h5py.Dataset) and name.endswith("autocorrelator"):
                parts = name.split("/")
                pi = parts.index("placements")
                envs.add(parts[pi + 2])
        root.visititems(visit)
    return envs


def _assert_dats_identical(single_dir, merged_dir, label, fails):
    s = _collect_dats(os.path.join(single_dir, "byo_dat"))
    m = _collect_dats(os.path.join(merged_dir, "byo_dat"))
    if not s:
        fails.append(f"[{label}] single-node byo_dat is empty — cannot compare")
        return
    if set(s) != set(m):
        fails.append(
            f"[{label}] byo_dat path-set differs: "
            f"only-single={sorted(set(s) - set(m))[:4]} "
            f"only-merged={sorted(set(m) - set(s))[:4]}"
        )
    for rel in sorted(set(s) & set(m)):
        if s[rel] != m[rel]:
            fails.append(f"[{label}] .dat bytes differ at {rel}")


def _assert_h5_subtree_equal(single_h5, merged_h5, root, label, fails):
    import numpy as np
    ds_s, at_s = _collect_h5_subtree(single_h5, root)
    ds_m, at_m = _collect_h5_subtree(merged_h5, root)
    # Empty-subtree guard, ROOT-PARAMETRIC (RED-RULING-ITEM3 §3 condition): a run
    # that produced no records must FAIL, not pass vacuously on an empty path-set
    # intersection — the guard travels with the root, not just the root string.
    if not at_s:
        fails.append(f"[{label}] single-node sweep.h5 has no /{root} subtree")
        return
    if set(at_s) != set(at_m):
        fails.append(
            f"[{label}] /{root} path-set differs: "
            f"only-single={sorted(set(at_s) - set(at_m))[:4]} "
            f"only-merged={sorted(set(at_m) - set(at_s))[:4]}"
        )
        return
    for name in sorted(ds_s):
        if not np.array_equal(ds_s[name], ds_m[name]):
            fails.append(f"[{label}] dataset values differ at {name}")
    for name in sorted(at_s):
        if at_s[name] != at_m[name]:
            fails.append(
                f"[{label}] attrs differ at {name}: {at_s[name]} != {at_m[name]}"
            )


# ── orchestration ───────────────────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description="Cross-node fan-out byte-identity gate.")
    ap.add_argument("--path", choices=("byo", "battery"), default="byo",
                    help="which sweep path to gate (selects fixture, reducer, "
                         "and HDF5 subtree root)")
    ap.add_argument("--config", default=None,
                    help="override fixture (default: the --path fixture)")
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    cfg_path = args.config or _FIXTURES[args.path]
    root = _H5_ROOT[args.path]

    if args.run_one:
        # Single engine invocation; shard env (if any) is inherited from the
        # parent's subprocess environment. The engine dispatches BYO vs battery
        # by experiment type, so this branch is path-agnostic.
        from lumi_hpc_qc.sweep.sweep_engine import run_sweep_from_yaml
        run_sweep_from_yaml(cfg_path)
        return 0

    import tempfile
    workdir = os.path.abspath(
        args.workdir or tempfile.mkdtemp(prefix="fanout_gate_")
    )
    os.makedirs(workdir, exist_ok=True)
    base = _load_cfg(cfg_path)
    seeds = sorted({
        int(s)
        for e in base["sweep"]["experiments"]
        for s in e.get("seed_list", [])
    })
    print(f"gate: path={args.path} fixture={cfg_path} root=/{root} "
          f"seeds={seeds} workdir={workdir}")

    fails = []

    # single-node baseline
    single_dir = os.path.join(workdir, "single")
    cfg_single = os.path.join(workdir, "cfg_single.yaml")
    _write_cfg(base, single_dir, cfg_single)
    print("gate: single-node baseline run ...")
    _spawn_run(cfg_single, shard=False)

    for nranks in (2, 3):
        d = os.path.join(workdir, f"r{nranks}")
        cfg = os.path.join(workdir, f"cfg_r{nranks}.yaml")
        _write_cfg(base, d, cfg)
        print(f"gate: {nranks}-rank shard runs ...")
        for r in range(nranks):
            _spawn_run(cfg, shard=True, nranks=nranks, rank=r)

        # §3.3 env-co-residency (BYO only): at the env-period-aligned divisor
        # (2), each rank must hold BOTH envs (the stratified shard's whole
        # point). The battery path runs all envs INSIDE each (seed, placement)
        # unit, so co-residency is structural, not a shard property — there is
        # nothing a shard could segregate, so this check does not apply
        # (RED-RULING-ITEM3 §3).
        if args.path == "byo" and nranks == 2:
            for r in range(nranks):
                envs = _envs_in_rank(os.path.join(d, f"sweep_rank{r}.h5"))
                if envs != _ENVS:
                    fails.append(
                        f"[2-rank §3.3] rank{r} envs={sorted(envs)} "
                        f"(expected both) — env co-residency broken"
                    )

        print(f"gate: merge {nranks}-rank shards ...")
        _merge(d, seeds, args.path)
        # .dat byte-identity is the BYO aggregation artifact; the battery path
        # has no .dat (identity reduce), so only the .h5 subtree is compared.
        if args.path == "byo":
            _assert_dats_identical(single_dir, d, f"{nranks}-rank .dat", fails)
        _assert_h5_subtree_equal(
            os.path.join(single_dir, "sweep.h5"),
            os.path.join(d, "sweep.h5"),
            root, f"{nranks}-rank .h5", fails,
        )

    if fails:
        print("\nGATE FAILED:")
        for fl in fails:
            print("  -", fl)
        return 1
    if args.path == "byo":
        detail = (".dat raw bytes + /byo dataset/path-set/attr equality; "
                  "env co-residency held at 2-rank")
    else:
        detail = ("/devices dataset/path-set/attr equality (identity reduce — "
                  "no .dat; envs co-resident by construction)")
    print(
        f"\nGATE PASSED [{args.path}]: single vs 2-rank vs 3-rank byte-identical "
        f"({detail}); seeds={seeds}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
