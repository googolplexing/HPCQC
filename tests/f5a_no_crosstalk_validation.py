#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""F5a no-cross-talk CI integration test (BLUE-PROPOSAL-F5A-NO-CROSSTALK-CI;
RED-RULING-F5A-NO-CROSSTALK-CI).

Promotes the Piece-3 hand diff (multi-placement device-cal subtrees ==
the isolated single-placement runs) to a standing, enforced guard. The F5a
lift was approved on the per-placement no-cross-talk evidence: in a
multi-placement device-cal sweep, each placement's noise model is composed
from its OWN qubits, so a multi-placement subtree is byte-identical to that
placement run in isolation. This mechanizes that proof.

Scale-independent: the composition path takes only (calibration,
physical_qubits, t2_mode) — it never sees the grid — so per-placement
composition is per-placement at any seed/kick/shot count. We therefore run
TINY (2 seeds x 6 kicks x 100 shots). This is the STRUCTURAL guard, not the
physics record; it does not reproduce z=60.7 (that stays the banked full-scale
run). Autocorr observable only: the lift is observable-general (the composition
path is identical for build_circuit / build_circuit_echo), so autocorr
certifies the path echo uses.

Three sweeps on cal 08c3c70f, two disjoint Piece-2 chains (HIGH/LOW),
noise_configs [noiseless, device_calibrated]:
  M  — ONE experiment, physical_qubits [[HIGH],[LOW]]  (the lifted multi-placement)
  H  — ONE experiment, physical_qubits [[HIGH]]         (HIGH in isolation)
  L  — ONE experiment, physical_qubits [[LOW]]          (LOW in isolation)

Checks, IN ORDER (the inventory guard runs FIRST so a silently-dropped
placement fails loudly instead of a missing-file diff passing vacuously):
  0. RECORD INVENTORY — M produced 2 placements x 2 envs x 2 seeds (8 leaves),
     both HIGH and LOW device-cal .dat present on disk.
  1. M-HIGH device-cal .dat == isolated H .dat            (no cross-talk, HIGH)
  2. M-LOW  device-cal .dat == isolated L .dat            (no cross-talk, LOW)
  3. noiseless byte-identical across placements (M-HIGH == M-LOW == H == L)
  4. noise_placement_independent flag: False on M device-cal (2 placements),
     True on H/L device-cal (1 placement), False on all noiseless.
  + HIGH != LOW device-cal (the 3.3x T2 contrast MUST show; guards against
    composition silently going placement-INDEPENDENT, which 1/2 would not catch).

Run on LUMI (needs qiskit-aer + h5py):
    srun ... python3 tests/f5a_no_crosstalk_validation.py
Expected: F5A NO-CROSSTALK: ALL CHECKS PASSED  (exit 0; exit 1 on any failure)
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

# ── Fixed setup: the two disjoint Piece-2 chains + the validated q10 disorder ──
HIGH = ["QB8", "QB16", "QB15", "QB23", "QB24", "QB25", "QB17", "QB18", "QB26", "QB27"]
LOW = ["QB34", "QB35", "QB43", "QB49", "QB50", "QB54", "QB53", "QB52", "QB48", "QB47"]
_CAL = os.path.join(project_dir, "examples", "q50_calibration_20260524_08c3c70f.json")
_DISORDER = os.path.join(project_dir, "examples", "byo", "floquet_disorder_q10_echo_ak10.json")
_SCRIPT = os.path.join(project_dir, "examples", "byo", "floquet_dtc_echo.py")
_SCRIPT_STEM = "floquet_dtc_echo"
SEEDS = [0, 1]
NUM_KICKS = 6       # range [0, 6) -> kicks 0..5
SHOTS = 100

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        failed += 1


def _experiment(placements):
    """One BYO autocorr experiment carrying a list of placements."""
    return {
        "type": "byo_circuit",
        "label": "f5a_crosstalk",
        "circuit_script": _SCRIPT,
        "circuit_function": "build_circuit",        # autocorr; lift is observable-general
        "fixed": {"num_qubits": 10, "epsilon": 0.03},
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
            "output_dir": tempfile.mkdtemp(prefix=f"f5a_xtalk_{sweep_id}_"),
            "sweep_id": sweep_id,
        }
    }


def _dat_path(out_dir, chain, env):
    """byo_dat/{stem}/{phys-qubits}/{env}/aggregated_autocorr.dat — single autocorr
    family, so byo_observable_subpath is "" (no circuit_function segment)."""
    phys = "-".join(chain)
    return os.path.join(out_dir, "byo_dat", _SCRIPT_STEM, phys, env,
                        "aggregated_autocorr.dat")


def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _first_diff(a, b):
    """(offset, a-line, b-line) of the first differing byte, for diagnostics."""
    n = min(len(a), len(b))
    off = next((i for i in range(n) if a[i] != b[i]), n)
    if off == n and len(a) == len(b):
        return None
    # decode a small window around the offset to a human-readable line pair
    def line_at(buf):
        s = buf[:off].rfind(b"\n") + 1
        e = buf.find(b"\n", off)
        e = e if e != -1 else len(buf)
        return buf[s:e].decode("utf-8", "replace")
    return off, line_at(a), line_at(b)


def _walk_flags(hdf5_path):
    """Return list of (noise_source, placement_phys_str, noise_placement_independent)
    for every BYO result leaf."""
    import h5py
    rows = []

    def visit(name, obj):
        if isinstance(obj, h5py.Group) and "noise_source" in obj.attrs:
            phys = ""
            if "physical_qubit_set" in obj:
                phys = "-".join(str(q) for q in obj["physical_qubit_set"][()])
            rows.append((
                str(obj.attrs.get("noise_source", "?")),
                phys,
                bool(obj.attrs["noise_placement_independent"])
                if "noise_placement_independent" in obj.attrs else None,
            ))

    with h5py.File(hdf5_path, "r") as f:
        f.visititems(visit)
    return rows


def main():
    from pathlib import Path
    from lumi_hpc_qc.sweep.sweep_engine import run_sweep_from_dict

    for p in (_CAL, _DISORDER, _SCRIPT):
        if not os.path.exists(p):
            print(f"F5A NO-CROSSTALK: MISSING INPUT {p}")
            return 1

    print("== Running three sweeps (M multi-placement, H/L singles) ==")
    res_m = run_sweep_from_dict(_sweep([HIGH, LOW], "f5a_xtalk_M"), device="CPU")
    res_h = run_sweep_from_dict(_sweep([HIGH], "f5a_xtalk_H"), device="CPU")
    res_l = run_sweep_from_dict(_sweep([LOW], "f5a_xtalk_L"), device="CPU")

    out_m = str(Path(res_m.hdf5_path).parent)
    out_h = str(Path(res_h.hdf5_path).parent)
    out_l = str(Path(res_l.hdf5_path).parent)

    # ── Check 0: RECORD INVENTORY (runs FIRST — closes the vacuous-pass hole) ──
    # M must have produced 2 placements x 2 envs x 2 seeds = 8 leaves, with BOTH
    # HIGH and LOW device-cal subtrees present on disk. If a regression dropped a
    # placement, a later byte-identity diff against a missing file could pass
    # vacuously; assert the thing-to-compare EXISTS before comparing it.
    m_flags = _walk_flags(res_m.hdf5_path)
    check("0a M leaf count == 8 (2 placements x 2 envs x 2 seeds)",
          len(m_flags) == 8, f"got {len(m_flags)} leaves: {m_flags}")
    m_dat_high = _dat_path(out_m, HIGH, "device_calibrated")
    m_dat_low = _dat_path(out_m, LOW, "device_calibrated")
    check("0b M HIGH device-cal .dat present", os.path.exists(m_dat_high), m_dat_high)
    check("0c M LOW device-cal .dat present", os.path.exists(m_dat_low), m_dat_low)
    m_dc_phys = {ph for (src, ph, _) in m_flags if src == "device_calibrated"}
    check("0d M device-cal covers BOTH placements",
          {"-".join(HIGH), "-".join(LOW)} <= m_dc_phys,
          f"device-cal placements present: {m_dc_phys}")

    # If the inventory is wrong, the byte-identity checks below are meaningless —
    # but we still run them (they will fail on the missing file, not pass) so the
    # report is complete. The inventory FAIL is the actionable signal.

    h_dat = _dat_path(out_h, HIGH, "device_calibrated")
    l_dat = _dat_path(out_l, LOW, "device_calibrated")

    # ── Checks 1 & 2: no-cross-talk byte-identity ──
    for name, multi, iso in (
        ("1 M-HIGH device-cal == isolated HIGH", m_dat_high, h_dat),
        ("2 M-LOW  device-cal == isolated LOW", m_dat_low, l_dat),
    ):
        if not (os.path.exists(multi) and os.path.exists(iso)):
            check(name, False, f"missing file: multi={multi} exists={os.path.exists(multi)}; "
                               f"iso={iso} exists={os.path.exists(iso)}")
            continue
        a, b = _read_bytes(multi), _read_bytes(iso)
        d = _first_diff(a, b)
        check(name, d is None,
              "byte-identical" if d is None
              else f"first diff at byte {d[0]}: multi={d[1]!r} iso={d[2]!r}")

    # ── Check 3: noiseless byte-identical across placements (the control) ──
    nz_paths = {
        "M-HIGH": _dat_path(out_m, HIGH, "noiseless"),
        "M-LOW": _dat_path(out_m, LOW, "noiseless"),
        "H": _dat_path(out_h, HIGH, "noiseless"),
        "L": _dat_path(out_l, LOW, "noiseless"),
    }
    missing = [k for k, p in nz_paths.items() if not os.path.exists(p)]
    if missing:
        check("3 noiseless byte-identical across placements", False,
              f"missing noiseless .dat for {missing}")
    else:
        ref = _read_bytes(nz_paths["M-HIGH"])
        mism = []
        for k in ("M-LOW", "H", "L"):
            d = _first_diff(ref, _read_bytes(nz_paths[k]))
            if d is not None:
                mism.append(f"{k}@byte{d[0]}")
        check("3 noiseless byte-identical across placements (M-HIGH==M-LOW==H==L)",
              not mism, f"mismatch vs M-HIGH: {mism}")

    # ── Check 4: flag truth, end-to-end on real sweeps ──
    def flag_ok(rows, src, want, where):
        got = [pi for (s, ph, pi) in rows if s == src]
        return all(pi is want for pi in got) and len(got) > 0, \
            f"{where} {src}: got {got}, want all {want}"

    ok_m_dc, d_m_dc = flag_ok(m_flags, "device_calibrated", False, "M")
    check("4a M device-cal noise_placement_independent == False (2 placements)",
          ok_m_dc, d_m_dc)
    ok_m_nz, d_m_nz = flag_ok(m_flags, "channels", False, "M")
    check("4b M noiseless noise_placement_independent == False", ok_m_nz, d_m_nz)

    h_flags = _walk_flags(res_h.hdf5_path)
    l_flags = _walk_flags(res_l.hdf5_path)
    ok_h, d_h = flag_ok(h_flags, "device_calibrated", True, "H")
    ok_l, d_l = flag_ok(l_flags, "device_calibrated", True, "L")
    check("4c H device-cal noise_placement_independent == True (1 placement)", ok_h, d_h)
    check("4d L device-cal noise_placement_independent == True (1 placement)", ok_l, d_l)

    # ── Red §4 add-1: HIGH != LOW device-cal (composition must be placement-DEPENDENT) ──
    if os.path.exists(m_dat_high) and os.path.exists(m_dat_low):
        check("+ HIGH device-cal != LOW device-cal (3.3x T2 contrast must show)",
              _read_bytes(m_dat_high) != _read_bytes(m_dat_low),
              "HIGH and LOW device-cal .dat are identical — composition went "
              "placement-INDEPENDENT (byte-identity-to-isolated would NOT catch this)")
    else:
        check("+ HIGH device-cal != LOW device-cal", False,
              "cannot compare — a device-cal .dat is missing (see check 0)")

    print()
    if failed == 0:
        print("F5A NO-CROSSTALK: ALL CHECKS PASSED")
        return 0
    print(f"F5A NO-CROSSTALK: {failed} CHECKS FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
