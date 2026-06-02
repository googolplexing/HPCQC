#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Workstream A §7.3 — placement-equivalence for the `disjoint` selection.

Reuses the F5a no-cross-talk byte-identity oracle (RED-RULING-WORKSTREAM-A §7.3:
"this is exactly the right reuse of the 44906bd CI guard"). Where the F5a CI
test pins HAND-PICKED HIGH/LOW chains, this pins the SOLVER-SELECTED disjoint
set: it proves the `disjoint` strategy's output is in the class the
no-cross-talk guarantee protects.

Flow:
  1. Run the solver's `disjoint` selection on cal 08c3c70f for the 10q chain to
     get K spatially-independent placements (K = the fidelity-ranked device-max;
     computed, printed, never assumed).
  2. Run those K as ONE multi-placement device-cal experiment (Run M), and each
     chain ci as an isolated single-placement experiment (Runs S_i).
  3. RECORD-INVENTORY guard FIRST (the F5a lesson): assert M produced all K
     placements x 2 envs x 2 seeds and every device-cal .dat is present — so the
     byte-identity diffs below cannot pass vacuously on a silently-dropped chain.
  4. For each ci: assert M's ci device-cal .dat is byte-identical to S_i's
     (the no-cross-talk proof, now over solver-selected chains).
  5. noiseless byte-identical across all placements (the control).
  6. flag truth: M device-cal records noise_placement_independent == False
     (K>1 placements); each S_i device-cal == True (1 placement).
  + the diversity provenance is banked (strategy/count_resolved/no_crosstalk).

TINY scale (2 seeds x 6 kicks x 100 shots): the no-cross-talk property is
scale-independent (the composition path never sees the grid), so this is the
structural guard, not a physics run.

Run on LUMI (needs qiskit-aer + h5py):
    srun ... python3 tests/f5a_diversity_equivalence_validation.py
Expected: WSA DIVERSITY EQUIVALENCE: ALL CHECKS PASSED  (exit 0; 1 on failure)
"""

import os
import sys
import tempfile

project_dir = os.environ.get(
    "PROJECT_DIR",
    os.environ.get(
        "SINGULARITYENV_PROJECT_DIR",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
)
sys.path.insert(0, os.path.join(project_dir, "src"))

_SCRIPT = os.path.join(project_dir, "examples", "byo", "floquet_dtc_echo.py")
_SCRIPT_STEM = "floquet_dtc_echo"
_CAL = os.path.join(
    project_dir, "examples", "q50_calibration_20260524_08c3c70f.json"
)
_DISORDER = os.path.join(project_dir, "examples", "byo", "q10_disorder.json")

SEEDS = [0, 1]
NUM_KICKS = 6        # range [0, 6) -> 0..5
SHOTS = 100
QSIZE = 10

passed = 0
failed = 0


def check(name, ok, detail=""):
    global passed, failed
    if ok:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        failed += 1


def _select_disjoint_chains():
    """Run the solver's disjoint selection -> list of qubit-name lists."""
    from lumi_hpc_qc.sweep.placement_solver import (
        GeneralPlacementSolver,
        select_disjoint_placements,
    )
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter

    cal = IQMv2Adapter().load(_CAL)
    solver = GeneralPlacementSolver()
    solver.add_device(cal)
    # 10q linear chain connectivity (logical i -- i+1).
    chain_edges = [(i, i + 1) for i in range(QSIZE - 1)]
    cands = solver.find_all_placements(
        circuit_edges=chain_edges, circuit_qubits=QSIZE, strategy="max_fidelity",
    )
    selected = select_disjoint_placements(cands, cal, count="auto", max_overlap=0)
    idx_to_name = cal.index_to_qubit_name
    chains = [
        [idx_to_name[i] for i in p.physical_indices] for p in selected
    ]
    return chains


def _experiment(placements):
    return {
        "type": "byo_circuit",
        "label": "wsa_diversity",
        "circuit_script": _SCRIPT,
        "circuit_function": "build_circuit",
        "fixed": {"num_qubits": QSIZE, "epsilon": 0.03},
        "grid": {"num_kicks": {"range": [0, NUM_KICKS]}},
        "disorder": {"source": "file", "file": _DISORDER, "initial_state": 3},
        "disorder_gates": ["rz", "rzz"],
        "physical_qubits": placements,
        "seed_list": SEEDS,
        "shots": SHOTS,
        "noise_configs": ["noiseless", "device_calibrated"],
    }


def _sweep(placements, sweep_id):
    return {
        "sweep": {
            "experiments": [_experiment(placements)],
            "calibrations": [_CAL],
            "output_dir": tempfile.mkdtemp(prefix=f"wsa_div_{sweep_id}_"),
            "sweep_id": sweep_id,
        }
    }


def _dat_path(out_dir, chain, env):
    phys = "-".join(chain)
    return os.path.join(out_dir, "byo_dat", _SCRIPT_STEM, phys, env,
                        "aggregated_autocorr.dat")


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _first_diff(a, b):
    n = min(len(a), len(b))
    off = next((i for i in range(n) if a[i] != b[i]), n)
    if off == n and len(a) == len(b):
        return None
    return off


def main():
    from lumi_hpc_qc.sweep.sweep_engine import run_sweep_from_dict

    print("=== Workstream A — disjoint placement-equivalence (F5a oracle) ===")

    # 1. Solver-selected disjoint chains (device-max, computed).
    chains = _select_disjoint_chains()
    K = len(chains)
    print(f"== disjoint selection: {K} chain(s) (fidelity-ranked device-max) ==")
    for i, c in enumerate(chains):
        print(f"   c{i}: {'-'.join(c)}")
    if K < 2:
        check("selection produced >=2 disjoint chains", False,
              f"got {K}; equivalence needs a multi-placement run")
        print("\nWSA DIVERSITY EQUIVALENCE: 1 FAILURE(S)")
        return 1

    # 2. Multi-placement run M + isolated runs S_i.
    print("== running M (all chains, multi-placement) + isolated singles ==")
    res_m = run_sweep_from_dict(_sweep(chains, "wsa_div_M"), device="CPU")
    out_m = res_m["output_dir"] if isinstance(res_m, dict) else res_m
    singles = []
    for i, c in enumerate(chains):
        res_s = run_sweep_from_dict(_sweep([c], f"wsa_div_S{i}"), device="CPU")
        singles.append(res_s["output_dir"] if isinstance(res_s, dict) else res_s)

    # 3. RECORD-INVENTORY guard FIRST (vacuous-pass hole, F5a §4(4) lesson).
    expect_leaves = K * 2 * len(SEEDS)   # K placements x 2 envs x 2 seeds
    present_dc = [os.path.exists(_dat_path(out_m, c, "device_calibrated"))
                  for c in chains]
    present_nl = [os.path.exists(_dat_path(out_m, c, "noiseless"))
                  for c in chains]
    check("0a M device-cal .dat present for ALL chains",
          all(present_dc), f"missing: {[chains[i] for i,p in enumerate(present_dc) if not p]}")
    check("0b M noiseless .dat present for ALL chains",
          all(present_nl), f"missing: {[chains[i] for i,p in enumerate(present_nl) if not p]}")
    check("0c M covers all K placements", all(present_dc) and all(present_nl),
          f"expected {expect_leaves} leaves across {K} chains")

    # 4. Per-chain byte-identity: M's ci device-cal == isolated S_i.
    for i, c in enumerate(chains):
        m_dat = _dat_path(out_m, c, "device_calibrated")
        s_dat = _dat_path(singles[i], c, "device_calibrated")
        if not (os.path.exists(m_dat) and os.path.exists(s_dat)):
            check(f"1.{i} M-c{i} device-cal == isolated S{i}", False,
                  "a .dat is missing (inventory should have caught this)")
            continue
        mb, sb = _read_bytes(m_dat), _read_bytes(s_dat)
        d = _first_diff(mb, sb)
        check(f"1.{i} M-c{i} device-cal == isolated S{i} (no cross-talk)",
              d is None, f"first byte diff at offset {d}")

    # 5. noiseless byte-identical across all placements + isolated.
    nl_ref = _read_bytes(_dat_path(out_m, chains[0], "noiseless"))
    nl_ok = True
    detail = ""
    for i, c in enumerate(chains):
        b_m = _read_bytes(_dat_path(out_m, c, "noiseless"))
        b_s = _read_bytes(_dat_path(singles[i], c, "noiseless"))
        if b_m != nl_ref or b_s != nl_ref:
            nl_ok = False
            detail = f"noiseless differs at chain c{i}"
            break
    check("3 noiseless byte-identical across placements (control)", nl_ok, detail)

    # 6. flag truth-table (reuse the F5a walker).
    from lumi_hpc_qc.sweep.sweep_engine import run_sweep_from_dict  # noqa: F401
    rows_m = _walk_flags(os.path.join(out_m, "sweep.h5"))
    dc_m = [r for r in rows_m if r[0] == "device_calibrated"]
    check("4a M device-cal noise_placement_independent == False (K>1)",
          len(dc_m) > 0 and all(r[2] is False for r in dc_m),
          f"flags={[r[2] for r in dc_m]}")
    for i in range(K):
        rows_s = _walk_flags(os.path.join(singles[i], "sweep.h5"))
        dc_s = [r for r in rows_s if r[0] == "device_calibrated"]
        check(f"4b.{i} S{i} device-cal noise_placement_independent == True (1)",
              len(dc_s) > 0 and all(r[2] is True for r in dc_s),
              f"flags={[r[2] for r in dc_s]}")

    # + diversity provenance banked + accurate.
    div = _walk_diversity(os.path.join(out_m, "sweep.h5"))
    check("+ diversity provenance banked (strategy=disjoint)",
          any(d.get("strategy") == "disjoint" for d in div),
          f"records={div[:1]}")
    check("+ count_resolved == K on the multi run",
          all(d.get("count_resolved") == K for d in div if d.get("strategy") == "disjoint"),
          f"count_resolved={[d.get('count_resolved') for d in div]}")
    check("+ no_crosstalk == True for max_overlap=0 run",
          all(d.get("no_crosstalk") is True for d in div if d.get("strategy") == "disjoint"),
          f"no_crosstalk={[d.get('no_crosstalk') for d in div]}")

    print()
    if failed:
        print(f"WSA DIVERSITY EQUIVALENCE: {failed} FAILURE(S)")
        return 1
    print("WSA DIVERSITY EQUIVALENCE: ALL CHECKS PASSED")
    return 0


def _walk_flags(hdf5_path):
    import h5py
    rows = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "noise_source" in obj.attrs:
            rows.append((
                str(obj.attrs.get("noise_source", "?")),
                "",
                bool(obj.attrs["noise_placement_independent"])
                if "noise_placement_independent" in obj.attrs else None,
            ))

    with h5py.File(hdf5_path, "r") as f:
        f.visititems(visit)
    return rows


def _walk_diversity(hdf5_path):
    """Pull the banked diversity provenance off every BYO record group."""
    import h5py
    out = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "placement_diversity_strategy" in obj.attrs:
            a = obj.attrs
            strat = a["placement_diversity_strategy"]
            out.append({
                "strategy": strat.decode() if isinstance(strat, bytes) else str(strat),
                "count_resolved": int(a.get("placement_diversity_count_resolved", -1)),
                "no_crosstalk": bool(a["placement_diversity_no_crosstalk"])
                if "placement_diversity_no_crosstalk" in a else None,
            })

    with h5py.File(hdf5_path, "r") as f:
        f.visititems(visit)
    return out


if __name__ == "__main__":
    sys.exit(main())
