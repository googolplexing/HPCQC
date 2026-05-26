# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""D3.4b — BYO counts observable + seed derivation.

The gate-2 reproduction depends on sweep.byo_observable being BYTE-IDENTICAL to
the banked repo-root floquet_runner_v2 (get_autocorrelation, resolve_instance_seed).
byo_observable re-homes them (rather than importing) because floquet_runner_v2 is
at the repo root, off the package path. These tests guard against drift by
importing BOTH and asserting equality on samples — if the bank changes and the
mirror doesn't, these fail.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from lumi_hpc_qc.sweep.byo_observable import (
    get_autocorrelation, resolve_instance_seed,
)

_REPO = Path(__file__).resolve().parents[2]


def _load_bank():
    """Import the repo-root floquet_runner_v2 by path (it is NOT on the package
    path), to compare the mirror against the source of truth."""
    bank_path = _REPO / "floquet_runner_v2.py"
    if not bank_path.exists():
        pytest.skip("floquet_runner_v2.py not at repo root")
    # floquet_runner_v2 imports lumi_hpc_qc.* at module import; ensure src is on
    # the path (the in-container PYTHONPATH already has it, but be explicit).
    src = str(_REPO / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    spec = importlib.util.spec_from_file_location("bank_frv2", bank_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bank_frv2"] = mod
    spec.loader.exec_module(mod)
    return mod


# ----------------- byte-identical-to-bank drift guard --------------------

def test_autocorrelation_matches_bank():
    bank = _load_bank()
    init = [0, 0, 0, 0]
    samples = [
        {"0000": 100},                         # all match -> +1
        {"1111": 100},                         # all mismatch -> -1
        {"0000": 50, "1111": 50},              # split -> 0
        {"0000": 70, "0011": 20, "1010": 10},  # mixed
    ]
    for counts in samples:
        assert get_autocorrelation(counts, init, 4) == bank.get_autocorrelation(
            counts, init, 4
        )


def test_resolve_instance_seed_matches_bank():
    bank = _load_bank()
    for ms in (0, 1234, 7):
        for inst in range(5):
            assert resolve_instance_seed(ms, inst) == bank.resolve_instance_seed(
                ms, inst
            )
    # entropy sentinel
    assert resolve_instance_seed(None, 0) is None
    assert resolve_instance_seed("random", 3) is None


# ------------------------- observable correctness ------------------------

def test_autocorrelation_known_values():
    init = [0, 0, 0, 0]
    assert get_autocorrelation({"0000": 100}, init, 4) == 1.0       # all match
    assert get_autocorrelation({"1111": 100}, init, 4) == -1.0      # all mismatch
    assert get_autocorrelation({"0000": 50, "1111": 50}, init, 4) == 0.0


def test_autocorrelation_little_endian_reversal():
    """Bitstring is reversed to little-endian before comparing to init_bit_array
    (Qiskit's count-key convention), matching the bank."""
    init = [1, 0, 0, 0]                         # qubit 0 expected = 1
    # bitstring "0001" reversed -> [1,0,0,0] -> all match -> +1
    assert get_autocorrelation({"0001": 100}, init, 4) == 1.0


def test_resolve_instance_seed_deterministic_and_distinct():
    a = resolve_instance_seed(0, 0)
    b = resolve_instance_seed(0, 1)
    assert a == resolve_instance_seed(0, 0)     # deterministic
    assert a != b                               # distinct per instance
    assert resolve_instance_seed(0, 0) != resolve_instance_seed(1, 0)  # per master


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
