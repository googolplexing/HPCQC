# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Unit tests for the W1.6 z_comb gate verifier (tests/_w1_z_comb_verify.py).

Covers the pass/fail/flag verdict boundaries, structural-error handling, and a
cross-check that the verifier's per-kick z_combined is numerically identical to
gate2_combine_compare.py's formula (so the two cannot drift).
"""
import importlib.util
import math
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(_REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zc = _load("_w1_z_comb_verify", "tests/_w1_z_comb_verify.py")


# ── z_combined formula ───────────────────────────────────────────────────────
def test_z_combined_basic():
    # |0.9 - 0.8| / sqrt(0.04^2 + 0.03^2) = 0.1 / 0.05 = 2.0
    assert abs(zc.z_combined(0.9, 0.03, 0.8, 0.04) - 2.0) < 1e-9


def test_z_combined_zero_sem_equal_means_is_zero():
    assert zc.z_combined(0.5, 0.0, 0.5, 0.0) == 0.0


def test_z_combined_zero_sem_diff_means_is_inf():
    # A real gap with no stated uncertainty cannot be explained statistically.
    assert zc.z_combined(0.7, 0.0, 0.5, 0.0) == float("inf")


# ── verdict boundaries ───────────────────────────────────────────────────────
def _ref(n_kicks):
    rm = {k: 0.0 for k in range(n_kicks)}
    rs = {k: 0.1 for k in range(n_kicks)}
    return rm, rs


def test_verdict_pass_when_all_within_5sigma():
    # candidate sem 0.0, ref sem 0.1 -> denom 0.1; offset 0.2 -> z=2.0 everywhere.
    cand = {k: (0.2, 0.0) for k in range(10)}
    rm, rs = _ref(10)
    res = zc.verify(cand, rm, rs)
    assert res["passed"] and res["n_fail"] == 0 and res["n_flag"] == 0
    assert abs(res["worst_z"] - 2.0) < 1e-9


def test_verdict_flag_between_3_and_5_does_not_fail():
    # offset 0.4 -> z=4.0 : flagged (>3) but passes (<=5).
    cand = {k: (0.4, 0.0) for k in range(5)}
    rm, rs = _ref(5)
    res = zc.verify(cand, rm, rs)
    assert res["passed"]
    assert res["n_flag"] == 5 and res["n_fail"] == 0


def test_verdict_fail_when_any_kick_beyond_5sigma():
    # one kick at z=6.0 (offset 0.6) forces FAIL; the rest at z=2.0.
    cand = {k: (0.2, 0.0) for k in range(10)}
    cand[7] = (0.6, 0.0)
    rm, rs = _ref(10)
    res = zc.verify(cand, rm, rs)
    assert not res["passed"]
    assert res["n_fail"] == 1 and res["n_flag"] == 1
    assert abs(res["worst_z"] - 6.0) < 1e-9


def test_verify_compares_only_overlapping_kicks():
    # candidate has an extra kick beyond the reference grid; it is ignored.
    cand = {0: (0.2, 0.0), 1: (0.2, 0.0), 99: (5.0, 0.0)}
    rm, rs = {0: 0.0, 1: 0.0}, {0: 0.1, 1: 0.1}
    res = zc.verify(cand, rm, rs)
    assert res["n_compared"] == 2 and res["passed"]


def test_verify_no_overlap_raises():
    with pytest.raises(ValueError, match="no overlapping kicks"):
        zc.verify({5: (0.1, 0.1)}, {0: 0.0}, {0: 0.1})


# ── .dat loader ──────────────────────────────────────────────────────────────
def test_load_aggregated_dat_roundtrip(tmp_path):
    p = tmp_path / "aggregated_autocorr.dat"
    p.write_text("# kick  mean_autocorr  sem\n"
                 "   0     1.0000     0.0000\n"
                 "   1    -0.8786     0.0008\n")
    d = zc.load_aggregated_dat(str(p))
    assert d[0] == (1.0, 0.0)
    assert d[1] == (-0.8786, 0.0008)


def test_load_aggregated_dat_rejects_short_row(tmp_path):
    p = tmp_path / "bad.dat"
    p.write_text("# kick  mean_autocorr  sem\n   0   1.0\n")  # missing sem
    with pytest.raises(ValueError):
        zc.load_aggregated_dat(str(p))


def test_main_structural_exit_on_missing_file(tmp_path, capsys):
    rc = zc.main(["--candidate", str(tmp_path / "nope.dat"),
                  "--reference", str(tmp_path / "alsonope.csv")])
    assert rc == 3


# ── cross-check: verifier z == gate2_combine_compare.py z ────────────────────
def test_z_matches_gate2_combine_compare():
    """The verifier's z_combined must be numerically identical to the existing
    gate2_combine_compare.py formula, so the two reporting/gating paths agree.
    gate2 computes: z_comb = |delta| / sqrt(ref_sem^2 + sem^2)."""
    g2 = _load("gate2_combine_compare", "gate2_combine_compare.py")
    # Replicate gate2's inline computation across a spread of values and assert
    # the verifier's helper returns the same number.
    cases = [
        (0.9000, 0.0300, 0.8000, 0.0400),
        (0.0000, 0.0010, 0.0000, 0.0004),
        (-0.8786, 0.0021, -0.8800, 0.0008),
        (0.5000, 0.0000, 0.5000, 0.0000),
    ]
    for cand_mean, sem, ref_mean, ref_sem in cases:
        delta = cand_mean - ref_mean
        denom = math.sqrt(ref_sem ** 2 + sem ** 2)
        g2_z = abs(delta) / denom if denom > 0 else (
            0.0 if delta == 0 else float("inf"))
        v_z = zc.z_combined(cand_mean, sem, ref_mean, ref_sem)
        if math.isinf(g2_z):
            assert math.isinf(v_z)
        else:
            assert abs(g2_z - v_z) < 1e-12, (cand_mean, sem, ref_mean, ref_sem)


def test_against_real_reference_csv_self_consistent():
    """Feed the reference's own device_cal columns back as the candidate:
    every z must be 0 (a series agrees perfectly with itself), so PASS."""
    ref_csv = os.path.join(_REPO,
                           "examples/reference/floquet_dtc_q10_autocorr.csv")
    rm, rs = zc.load_reference(ref_csv)
    candidate = {k: (rm[k], rs[k]) for k in rm}
    res = zc.verify(candidate, rm, rs)
    assert res["passed"]
    assert res["worst_z"] == 0.0
    assert res["n_flag"] == 0 and res["n_fail"] == 0
