# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""NF3 — hardened W1 canary verifier (tests/_w1_canary_verify.py).

Pure-stdlib, offline-runnable (no qiskit/aer/h5py): builds synthetic
``instance_NN_autocorr.dat`` trees with controlled bytes + a synthetic oracle,
runs the verifier as a subprocess, and asserts the acceptance taxonomy.

The load-bearing case is ``test_arm_label_swap_now_fails``: the old
"PASS iff any arm matches" bar would have ACCEPTED device-cal physics written
under a ``noiseless/`` directory (an arm-label swap). The hardened verifier
must reject it — that is the regression NF3 elevated from Blue DEBT #1.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

VERIFIER = Path(__file__).resolve().parents[1] / "_w1_canary_verify.py"

# Two seeds, as in the 2-seed canary.
_SEEDS = (0, 1)
_DEVICE_CAL_BYTES = {
    0: b"device_calibrated seed 0 autocorrelator payload\n",
    1: b"device_calibrated seed 1 autocorrelator payload\n",
}
_NOISELESS_BYTES = {
    0: b"noiseless seed 0 autocorrelator payload (distinct)\n",
    1: b"noiseless seed 1 autocorrelator payload (distinct)\n",
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _write_dat(workdir: Path, arm: str, seed_idx: int, payload: bytes) -> None:
    d = workdir / "sweep_output" / "w1_canary" / "byo_dat" / "floquet" / "QB11" / arm
    d.mkdir(parents=True, exist_ok=True)
    (d / f"instance_{seed_idx:02d}_autocorr.dat").write_bytes(payload)


def _write_oracle(path: Path, payloads: dict[int, bytes]) -> None:
    # sha256sum format; the parser keys on a ``seed_NN_instance`` path segment.
    lines = [
        f"{_sha(payloads[s])}  evidence/W1/gate2_canary/canary_seed_{s:02d}_instance.dat"
        for s in sorted(payloads)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(workdir: Path, oracle: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(VERIFIER),
         "--workdir", str(workdir), "--oracle", str(oracle)],
        capture_output=True, text=True,
    )


def _make_oracle(tmp_path: Path) -> Path:
    oracle = tmp_path / "sha256_oracle.txt"
    _write_oracle(oracle, _DEVICE_CAL_BYTES)
    return oracle


def test_pass_when_devicecal_matches_and_noiseless_distinct(tmp_path):
    wd = tmp_path / "wd"
    for s in _SEEDS:
        _write_dat(wd, "device_calibrated", s, _DEVICE_CAL_BYTES[s])
        _write_dat(wd, "noiseless", s, _NOISELESS_BYTES[s])
    r = _run(wd, _make_oracle(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    assert "ALL CHECKS PASSED" in r.stdout


def test_fail_when_devicecal_mismatches(tmp_path):
    wd = tmp_path / "wd"
    _write_dat(wd, "device_calibrated", 0, _DEVICE_CAL_BYTES[0])
    _write_dat(wd, "device_calibrated", 1, b"corrupted device-cal seed 1\n")
    for s in _SEEDS:
        _write_dat(wd, "noiseless", s, _NOISELESS_BYTES[s])
    r = _run(wd, _make_oracle(tmp_path))
    assert r.returncode == 1
    assert "device_calibrated arm does NOT byte-match" in r.stdout


def test_fail_when_devicecal_missing(tmp_path):
    wd = tmp_path / "wd"
    for s in _SEEDS:
        _write_dat(wd, "noiseless", s, _NOISELESS_BYTES[s])
    r = _run(wd, _make_oracle(tmp_path))
    assert r.returncode == 1
    assert "device_calibrated arm is MISSING" in r.stdout


def test_fail_when_noiseless_missing(tmp_path):
    wd = tmp_path / "wd"
    for s in _SEEDS:
        _write_dat(wd, "device_calibrated", s, _DEVICE_CAL_BYTES[s])
    r = _run(wd, _make_oracle(tmp_path))
    assert r.returncode == 1
    assert "noiseless arm is MISSING" in r.stdout


def test_fail_when_noiseless_matches_devicecal_oracle(tmp_path):
    # Noiseless arm reproduces the device-cal oracle bytes -> noise not applied.
    wd = tmp_path / "wd"
    for s in _SEEDS:
        _write_dat(wd, "device_calibrated", s, _DEVICE_CAL_BYTES[s])
        _write_dat(wd, "noiseless", s, _DEVICE_CAL_BYTES[s])
    r = _run(wd, _make_oracle(tmp_path))
    assert r.returncode == 1
    assert "noiseless arm MATCHES the device-cal oracle" in r.stdout


def test_arm_label_swap_now_fails(tmp_path):
    # The NF3 regression: device-cal physics written under noiseless/, and the
    # device_calibrated/ dir absent. The OLD "any arm matches" bar PASSED here
    # (by_arm["noiseless"] matched the oracle). The hardened bar must FAIL:
    # device-cal arm missing AND noiseless matching the device-cal oracle.
    wd = tmp_path / "wd"
    for s in _SEEDS:
        _write_dat(wd, "noiseless", s, _DEVICE_CAL_BYTES[s])
    r = _run(wd, _make_oracle(tmp_path))
    assert r.returncode == 1, (
        "arm-label swap must be rejected (this is the NF3 hole)"
    )
    assert "device_calibrated arm is MISSING" in r.stdout
    assert "noiseless arm MATCHES the device-cal oracle" in r.stdout
