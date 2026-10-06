# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Negative + positive test for the PRODUCTION merge CLI on the battery path
(RED-RULING-MERGE-CLI-FOLLOWUP §4 ask-3 / §5 matrix).

Runs the battery fixture as 2 shard ranks, then exercises the PRODUCTION CLI
(scripts/merge_sweep_shards.py — NOT the gate's internal _merge), twice:

  POSITIVE  — merge the COMPLETE shards: must exit 0, write sweep.h5, and select
              BatteryReducer by config type (the production reducer-selection
              path, RED-CLARIFICATION verification point 2).
  NEGATIVE  — drop ONE unit (one (seed,placement,env) record) from a shard, then
              merge: that (placement,env) group is now short one seed, so the
              completeness guard MUST fail loud (INCOMPLETE), non-zero exit
              (verification point 3).

A SINGLE-UNIT drop is used, NOT a whole-rank drop: at num_placements % nranks ==
0 a whole-rank drop removes a group ENTIRELY, and the present-group completeness
guard is BLIND to a wholly-absent group (the blind spot closed only by option
(i)). The single-unit drop leaves a group short, so the guard fires
deterministically, independent of ordering and nranks — it is the case that
proves the guard is live on /devices.

In-container (h5py + the engine). Single node; the 2 ranks are subprocesses with
HPCQC_SWEEP_SHARD=1 (completeness is allocation-shape-independent). Run via
tests/slurm_fanout_negative_dropunit.sh.
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


def _run_cli(out_dir):
    proc = subprocess.run(
        [sys.executable, "-u", _CLI,
         "--output-dir", out_dir, "--config", _FIXTURE],
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


def _drop_one_unit(shard_h5):
    """Delete exactly one (seed,placement,env) unit group (the leaf parent of an
    energy_trajectory dataset) so its (placement,env) group is short one seed."""
    import h5py

    with h5py.File(shard_h5, "r+") as f:
        victim = []

        def _visit(name, obj):
            if (not victim and isinstance(obj, h5py.Dataset)
                    and name.endswith("energy_trajectory")):
                victim.append(name)

        f.visititems(_visit)
        if not victim:
            raise SystemExit(
                f"drop: no energy_trajectory unit found in {shard_h5}"
            )
        parent = os.path.dirname(victim[0])  # the (seed,placement,env) leaf group
        del f[parent]
        return parent


def main():
    import tempfile

    workdir = os.path.abspath(
        os.environ.get("NEG_WORKDIR") or tempfile.mkdtemp(prefix="fanout_neg_")
    )
    run_dir = os.path.join(workdir, "run")
    print(f"neg-test: workdir={workdir}")
    print(f"neg-test: {_NRANKS} battery shard runs ...")
    for r in range(_NRANKS):
        _shard_run(run_dir, r, _NRANKS)

    fails = []

    # POSITIVE — complete shards through the production CLI
    pos = os.path.join(workdir, "pos")
    shutil.copytree(run_dir, pos)
    rc, out = _run_cli(pos)
    if rc != 0:
        fails.append(
            f"[positive] production CLI exited {rc} on COMPLETE shards:\n{out}"
        )
    elif not os.path.exists(os.path.join(pos, "sweep.h5")):
        fails.append("[positive] production CLI exit 0 but no sweep.h5 written")
    elif "BatteryReducer" not in out:
        fails.append(
            f"[positive] production CLI did not select BatteryReducer:\n{out}"
        )
    else:
        print("neg-test: POSITIVE ok — exit 0, sweep.h5 written, "
              "reducer=BatteryReducer")

    # NEGATIVE — drop one unit, expect INCOMPLETE fail-loud
    neg = os.path.join(workdir, "neg")
    shutil.copytree(run_dir, neg)
    shards = sorted(glob.glob(os.path.join(neg, "sweep_rank*.h5")))
    dropped = _drop_one_unit(shards[0])
    print(f"neg-test: dropped unit group {dropped!r} from "
          f"{os.path.basename(shards[0])}")
    rc, out = _run_cli(neg)
    if rc == 0:
        fails.append(
            "[negative] production CLI exited 0 after a unit drop — the "
            "completeness guard did NOT fire (vacuous pass)"
        )
    elif "INCOMPLETE" not in out:
        fails.append(f"[negative] CLI exited {rc} but not with INCOMPLETE:\n{out}")
    else:
        print(f"neg-test: NEGATIVE ok — exit {rc}, INCOMPLETE raised as expected")

    if fails:
        print("\nNEG-TEST FAILED:")
        for fl in fails:
            print("  -", fl)
        return 1
    print("\nNEG-TEST PASSED: production CLI selects BatteryReducer on a complete "
          "battery merge; a single-unit drop fails loud (INCOMPLETE).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
