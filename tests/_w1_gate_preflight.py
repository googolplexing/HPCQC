#!/usr/bin/env python3
"""W1.6 Option-1 gate pre-flight (RED ruling #3 / fold §4.1).

Fail loud BEFORE the expensive sweep if the gated run is not actually pinned:

  1. master_seed resolves to int 0. On the file-disorder path the engine reads
     the disorder file's ``_meta.master_seed`` and feeds it to
     ``seed_simulator = resolve_instance_seed(master_seed, seed)``
     (sweep_engine.py:487,551). A missing / None / "random" / non-int value
     silently sends Aer to entropy -- the worker falls back to
     ``int(inst_seed) if inst_seed is not None else 0`` (byo_worker) -- which
     makes BOTH arms non-reproducible and could "pass" the z_comb gate by
     accident (two random draws are statistically close). This guard makes that
     desync impossible to launch into.

  2. physical_qubits is pinned to exactly ONE well-formed q10 placement, i.e.
     the device-calibrated arm is NOT free-layout. The EXACT canonical identity
     is owned by tests/unit/test_canonical_placement_guard.py (the single
     recorded ``_CANONICAL`` authority); it is deliberately not re-hardcoded
     here, so this stays a structural pre-flight and there is one source of the
     canonical order.

  3. disorder.source is ``file`` and points at the same disorder file whose
     master_seed we just checked (otherwise the seed check would not apply to
     the file the sweep actually reads).

Reuses the engine's own ``_parse_physical_qubits`` so "pinned" means exactly
what the engine means by it -- no parallel parsing logic here.

Usage:
    python3 tests/_w1_gate_preflight.py <sweep.yaml> <disorder.json>

Exit 0 = OK; 1 = a check failed (message on stderr); 2 = bad invocation.
"""
from __future__ import annotations

import json
import sys

import yaml

from lumi_hpc_qc.sweep.sweep_engine import _parse_physical_qubits

_N_QUBITS = 10


def _fail(msg: str) -> int:
    print(f"[gate-preflight] FAIL: {msg}", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 2:
        print("usage: _w1_gate_preflight.py <sweep.yaml> <disorder.json>",
              file=sys.stderr)
        return 2
    cfg_path, dis_path = argv

    # ── 1. master_seed == int 0 (the value the file path feeds seed_simulator) ──
    try:
        with open(dis_path, encoding="utf-8") as f:
            dis = json.load(f)
    except (OSError, ValueError) as e:
        return _fail(f"cannot read disorder file {dis_path!r}: {e}")
    meta = dis.get("_meta")
    if not isinstance(meta, dict) or "master_seed" not in meta:
        return _fail(f"{dis_path}: missing _meta.master_seed")
    ms = meta["master_seed"]
    # bool is an int subclass -- exclude True/False explicitly.
    if isinstance(ms, bool) or not isinstance(ms, int):
        return _fail(
            f"{dis_path}: _meta.master_seed must be an int, got {ms!r} "
            f"({type(ms).__name__}); a non-int/None resolves to entropy and "
            f"breaks reproducibility of seed_simulator")
    if ms != 0:
        return _fail(
            f"{dis_path}: _meta.master_seed must be 0 for the gate, got {ms}")

    # ── 2 & 3. config: pinned single placement + file-disorder consistency ──
    try:
        with open(cfg_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        return _fail(f"cannot read sweep config {cfg_path!r}: {e}")
    exps = (cfg.get("sweep") or {}).get("experiments") or []
    if len(exps) != 1:
        return _fail(
            f"{cfg_path}: expected exactly one experiment, got {len(exps)}")
    exp = exps[0]

    try:
        pinned = _parse_physical_qubits(exp.get("physical_qubits"))
    except ValueError as e:
        return _fail(f"{cfg_path}: physical_qubits malformed: {e}")
    if pinned is None:
        return _fail(
            f"{cfg_path}: physical_qubits is not pinned -- the "
            f"device-calibrated arm would free-layout; Option-1 requires a "
            f"pinned canonical placement")
    if len(pinned) != 1:
        return _fail(
            f"{cfg_path}: Option-1 pins exactly ONE placement, got {len(pinned)}")
    placement = pinned[0]
    if len(placement) != _N_QUBITS or len(set(placement)) != _N_QUBITS:
        return _fail(
            f"{cfg_path}: pinned placement must be {_N_QUBITS} distinct qubit "
            f"names, got {placement}")

    dis_cfg = exp.get("disorder") or {}
    if dis_cfg.get("source") != "file":
        return _fail(
            f"{cfg_path}: gate requires disorder.source: file, got "
            f"{dis_cfg.get('source')!r}")
    if dis_cfg.get("file") != dis_path:
        return _fail(
            f"{cfg_path}: disorder.file {dis_cfg.get('file')!r} != the "
            f"pre-flight disorder file {dis_path!r}; the seed check would not "
            f"apply to the file the sweep reads")

    print(f"[gate-preflight] OK: master_seed=0 (int); pinned 1x{_N_QUBITS} "
          f"placement {placement}; file disorder {dis_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
