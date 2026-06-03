#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Byte-identity GATE for the Workstream-B cross-node fan-out (BYO path).

Runs the 8-unit fixture three ways and proves the shard+merge is byte-identical
to a single-node run (RED-RULING-WORKSTREAM-B §6.1/§8.4;
RED-REVIEW-WORKSTREAM-B-INCREMENTS-1-2 §7.3):

  * single-node (HPCQC_SWEEP_SHARD unset) — the baseline;
  * 2-rank shard (the divisor that would segregate envs under plain round-robin
    — also the §3.3 env-co-residency check: each rank must hold BOTH envs);
  * 3-rank shard (8 units / 3 -> 3/3/2, the non-dividing case where a shard
    off-by-one would hide).

Each rank is a separate engine process with HPCQC_SWEEP_SHARD=1 +
SLURM_NNODES/SLURM_NODEID set, so the gate needs only ONE node (byte-identity is
allocation-shape-independent — a unit's output is a pure function of its seed).
After each shard run the gate merges the rank shards and asserts:

  (a) the byo_dat .dat tree is byte-identical to single-node (raw bytes — the
      certified comparison artifact, matching every prior gate: W1.6, no-cross-
      talk CI);
  (b) the merged sweep.h5 /byo subtree matches single-node at the DATASET level
      (NOT raw .h5 bytes — HDF5 container layout legitimately differs between one
      writer and a union): identical group/dataset path-set, exact np.array_equal
      on every dataset, and equal per-node attributes (noise_placement_independent,
      calibration_set_id, seed, ...). Run-level metadata attrs (created_at, etc.)
      are not compared — they are not physics.

Exit 0 on pass, 1 on any mismatch. Requires the container (qiskit-aer + h5py).

Usage (inside the container, from the repo root):
  python3 tests/fanout_byte_identity_gate.py
  python3 tests/fanout_byte_identity_gate.py --config <fixture.yaml> --workdir <dir>
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

_DEFAULT_FIXTURE = "examples/byo/floquet_dtc_q10_fanout_gate_8unit.yaml"
_ENVS = {"noiseless", "device_calibrated"}


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


def _merge(output_dir, seeds):
    import glob
    from lumi_hpc_qc.sweep.fanout_merge import ByoAutocorrReducer, merge_shards
    rank_h5s = sorted(glob.glob(os.path.join(output_dir, "sweep_rank*.h5")))
    manifests = sorted(
        glob.glob(os.path.join(output_dir, "campaign_manifest_rank*.json"))
    )
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


def _collect_byo_h5(h5path):
    import h5py
    import numpy as np
    ds_vals, node_attrs = {}, {}
    with h5py.File(h5path, "r") as f:
        root = f.get("byo")
        if root is None:
            return ds_vals, node_attrs

        def visit(name, obj):
            node_attrs[name] = {k: _norm_attr(v) for k, v in obj.attrs.items()}
            if isinstance(obj, h5py.Dataset):
                ds_vals[name] = np.asarray(obj[()])
        root.visititems(visit)
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


def _assert_h5_byo_equal(single_h5, merged_h5, label, fails):
    import numpy as np
    ds_s, at_s = _collect_byo_h5(single_h5)
    ds_m, at_m = _collect_byo_h5(merged_h5)
    if not at_s:
        fails.append(f"[{label}] single-node sweep.h5 has no /byo subtree")
        return
    if set(at_s) != set(at_m):
        fails.append(
            f"[{label}] /byo path-set differs: "
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
    ap.add_argument("--config", default=_DEFAULT_FIXTURE)
    ap.add_argument("--workdir", default=None)
    ap.add_argument("--run-one", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    if args.run_one:
        # Single engine invocation; shard env (if any) is inherited from the
        # parent's subprocess environment.
        from lumi_hpc_qc.sweep.sweep_engine import run_sweep_from_yaml
        run_sweep_from_yaml(args.config)
        return 0

    import tempfile
    workdir = os.path.abspath(
        args.workdir or tempfile.mkdtemp(prefix="fanout_gate_")
    )
    os.makedirs(workdir, exist_ok=True)
    base = _load_cfg(args.config)
    seeds = sorted({
        int(s)
        for e in base["sweep"]["experiments"]
        for s in e.get("seed_list", [])
    })
    print(f"gate: fixture={args.config} seeds={seeds} workdir={workdir}")

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

        # §3.3 env-co-residency: at the env-period-aligned divisor (2), each
        # rank must hold BOTH envs (the stratified shard's whole point).
        if nranks == 2:
            for r in range(nranks):
                envs = _envs_in_rank(os.path.join(d, f"sweep_rank{r}.h5"))
                if envs != _ENVS:
                    fails.append(
                        f"[2-rank §3.3] rank{r} envs={sorted(envs)} "
                        f"(expected both) — env co-residency broken"
                    )

        print(f"gate: merge {nranks}-rank shards ...")
        _merge(d, seeds)
        _assert_dats_identical(single_dir, d, f"{nranks}-rank .dat", fails)
        _assert_h5_byo_equal(
            os.path.join(single_dir, "sweep.h5"),
            os.path.join(d, "sweep.h5"),
            f"{nranks}-rank .h5", fails,
        )

    if fails:
        print("\nGATE FAILED:")
        for fl in fails:
            print("  -", fl)
        return 1
    print(
        "\nGATE PASSED: single vs 2-rank vs 3-rank byte-identical "
        "(.dat raw bytes + /byo dataset/path-set/attr equality); "
        f"env co-residency held at 2-rank; seeds={seeds}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
