# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Acceptance gate for option (i) — the WHOLLY-ABSENT-group guard — on the
PRODUCTION merge CLI (RED-RULING-PATCH43-VERIFY-AND-INVENTORY-DESIGN §3 Q4, §4.3).

Runs the battery fixture as 2 shard ranks (which makes the engine write
campaign_expected.json), then exercises the PRODUCTION CLI
(scripts/merge_sweep_shards.py — NOT the gate's internal _merge) three ways, all
with --nranks present (the production configuration):

  POSITIVE — COMPLETE shards + the inventory present, --nranks=NRANKS: must exit
             0, write sweep.h5, select BatteryReducer, and the inventory group-set
             check must run SILENTLY (the merge prints "group-set checked vs N
             expected"). This is the end-to-end proof that the engine's expected
             keys MATCH the keys extract yields from the written paths — i.e. the
             single-source-of-truth has no drift on real data. (Byte-identity vs a
             single-node run is the separate byte-identity gate's job.)
  (i-a)    — drop a whole RANK FILE (one sweep_rank*.h5), --nranks=NRANKS: must
             fail loud at DISCOVERY (file count != nranks), before the union runs.
             Proves the cheap missing-file guard.
  (i-b)    — drop a whole (placement,env) GROUP (ALL its seeds, wherever they
             live across shards) from WITHIN present files, all NRANKS files
             present, --nranks=NRANKS: must fail loud at the GROUP-SET assert
             ("missing" names the absent group). This is THE blind-spot closure —
             the residual case --nranks cannot see, the direct analog of the
             single-unit-drop gate lifted from "group short one seed" to "group
             entirely absent".

A whole-GROUP drop (all seeds), NOT a single-unit drop: a single-unit drop only
makes a group SHORT (that is tests/fanout_negative_dropunit.py — the short-count
guard). Removing every seed of a group makes it WHOLLY ABSENT, which only the
inventory group-set check can catch.

In-container (h5py + the engine). Single node; the 2 ranks are subprocesses with
HPCQC_SWEEP_SHARD=1. Run via tests/slurm_fanout_negative_dropgroup.sh.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_FIXTURE = os.path.join(
    _REPO_ROOT, "examples", "battery", "tfim_4q_fanout_gate.yaml"
)
_CLI = os.path.join(_REPO_ROOT, "scripts", "merge_sweep_shards.py")
_NRANKS = 2


def _shard_run(out_dir, rank, nranks):
    os.makedirs(out_dir, exist_ok=True)
    env = dict(os.environ)
    env["HPCQC_SWEEP_SHARD"] = "1"
    env["SLURM_NNODES"] = str(nranks)
    env["SLURM_NODEID"] = str(rank)
    env.setdefault("PYTHONPATH", os.path.join(_REPO_ROOT, "src"))
    proc = subprocess.run(
        [sys.executable, "-u", "-m", "lumi_hpc_qc.sweep.run_sweep",
         _FIXTURE, "--output-dir", out_dir],
        env=env, capture_output=True, text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(f"shard run rank {rank}/{nranks} FAILED")


def _run_cli(out_dir, nranks=None):
    cmd = [sys.executable, "-u", _CLI,
           "--output-dir", out_dir, "--config", _FIXTURE]
    if nranks is not None:
        cmd += ["--nranks", str(nranks)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout + proc.stderr


def _group_key(name):
    # the unit's GROUP key via the single source of truth (dirname of the
    # energy_trajectory dataset) — identical to what extract + the inventory use.
    from lumi_hpc_qc.sweep.battery_paths import group_key_from_path
    return group_key_from_path(name.rsplit("/", 1)[0])


def _pick_target_group(shard_h5_paths):
    import h5py

    for sh in shard_h5_paths:
        found = []
        with h5py.File(sh, "r") as f:
            def _v(name, obj):
                if (not found and isinstance(obj, h5py.Dataset)
                        and name.endswith("energy_trajectory")):
                    found.append(name)
            f.visititems(_v)
        if found:
            return _group_key(found[0])
    raise SystemExit("drop-group: no energy_trajectory unit found in any shard")


def _drop_whole_group(shard_h5_paths, target):
    """Delete EVERY unit (any seed, in any shard file) whose group key == target,
    so the (placement,env[,params]) group is WHOLLY absent from the union while
    every shard FILE remains present."""
    import h5py

    removed = 0
    for sh in shard_h5_paths:
        with h5py.File(sh, "r+") as f:
            victims = []

            def _v(name, obj):
                if (isinstance(obj, h5py.Dataset)
                        and name.endswith("energy_trajectory")
                        and _group_key(name) == target):
                    victims.append(os.path.dirname(name))  # the unit leaf group

            f.visititems(_v)
            for grp in victims:
                if grp in f:
                    del f[grp]
                    removed += 1
    return removed


def main():
    import tempfile

    workdir = os.path.abspath(
        os.environ.get("NEG_WORKDIR") or tempfile.mkdtemp(prefix="fanout_dropgroup_")
    )
    run_dir = os.path.join(workdir, "run")
    print(f"dropgroup-gate: workdir={workdir}")
    print(f"dropgroup-gate: {_NRANKS} battery shard runs (writes "
          f"campaign_expected.json) ...")
    for r in range(_NRANKS):
        _shard_run(run_dir, r, _NRANKS)

    inv = os.path.join(run_dir, "campaign_expected.json")
    if not os.path.exists(inv):
        raise SystemExit(
            "dropgroup-gate: the shard runs did NOT write campaign_expected.json "
            "— the engine inventory generator did not fire (option (i) broken)."
        )
    print(f"dropgroup-gate: inventory present ({os.path.basename(inv)})")

    fails = []

    # ── POSITIVE — complete shards + inventory, --nranks present ──
    pos = os.path.join(workdir, "pos")
    shutil.copytree(run_dir, pos)
    rc, out = _run_cli(pos, nranks=_NRANKS)
    if rc != 0:
        fails.append(f"[positive] CLI exited {rc} on COMPLETE shards:\n{out}")
    elif not os.path.exists(os.path.join(pos, "sweep.h5")):
        fails.append("[positive] CLI exit 0 but no sweep.h5 written")
    elif "BatteryReducer" not in out:
        fails.append(f"[positive] CLI did not select BatteryReducer:\n{out}")
    elif "group-set checked vs" not in out:
        fails.append(
            f"[positive] inventory present but the group-set check did not run "
            f"(expected 'group-set checked vs N expected'):\n{out}"
        )
    else:
        print("dropgroup-gate: POSITIVE ok — exit 0, sweep.h5, BatteryReducer, "
              "group-set checked silently against the inventory")

    # ── (i-a) — drop a whole RANK FILE, --nranks present → discovery fail ──
    a = os.path.join(workdir, "drop_a")
    shutil.copytree(run_dir, a)
    a_shards = sorted(glob.glob(os.path.join(a, "sweep_rank*.h5")))
    victim_file = a_shards[-1]
    os.remove(victim_file)
    print(f"dropgroup-gate: (i-a) removed rank file "
          f"{os.path.basename(victim_file)} ({len(a_shards) - 1}/{_NRANKS} left)")
    rc, out = _run_cli(a, nranks=_NRANKS)
    if rc == 0:
        fails.append(
            "[i-a] CLI exited 0 after a whole-rank-file drop — the --nranks "
            "discovery guard did NOT fire"
        )
    elif "nranks" not in out and "(i-a)" not in out:
        fails.append(f"[i-a] CLI exited {rc} but not on the --nranks guard:\n{out}")
    else:
        print(f"dropgroup-gate: (i-a) ok — exit {rc}, --nranks discovery guard "
              "fired at file count")

    # ── (i-b) — drop a whole GROUP from present files, --nranks present →
    #    group-set assert fail (THE blind-spot closure) ──
    b = os.path.join(workdir, "drop_b")
    shutil.copytree(run_dir, b)
    b_shards = sorted(glob.glob(os.path.join(b, "sweep_rank*.h5")))
    target = _pick_target_group(b_shards)
    n_removed = _drop_whole_group(b_shards, target)
    still_present = sorted(glob.glob(os.path.join(b, "sweep_rank*.h5")))
    print(f"dropgroup-gate: (i-b) removed whole group {target!r} "
          f"({n_removed} unit(s) across {len(still_present)} present file(s))")
    if len(still_present) != _NRANKS:
        fails.append(
            f"[i-b] fixture error: {len(still_present)} files present, "
            f"expected {_NRANKS} (i-b must drop a GROUP, not a FILE)"
        )
    rc, out = _run_cli(b, nranks=_NRANKS)
    if rc == 0:
        fails.append(
            "[i-b] CLI exited 0 after a WHOLLY-ABSENT group — the group-set "
            "inventory guard did NOT fire (the blind spot is still open)"
        )
    elif "missing" not in out or "group SET mismatch" not in out:
        fails.append(
            f"[i-b] CLI exited {rc} but not on the group-set assert "
            f"(expected 'group SET mismatch ... missing'):\n{out}"
        )
    else:
        print(f"dropgroup-gate: (i-b) ok — exit {rc}, group-set assert fired "
              "(missing group named)")

    if fails:
        print("\nDROPGROUP-GATE FAILED:")
        for fl in fails:
            print("  -", fl)
        return 1
    print("\nDROPGROUP-GATE PASSED: complete+inventory merges silently; a "
          "whole-rank-file drop fails loud at discovery (i-a); a wholly-absent "
          "group fails loud at the group-set assert (i-b).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
