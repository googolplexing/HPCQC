# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Unit tests for the Step-1 systematic-residual mode of the z_comb verifier.

RED-CLARIFICATION-STEP1-SIGMA-SYS: Step 1 is a MEASUREMENT, not a 5-sigma gate.
This pins the relative-deviation metric (with the near-zero floor), the
even-kick decay-rate fit, the residual summary, and the convergence verdict
against the PRE-COMMITTED ceiling (which is distinct from the measured
sigma_sys). Same cross-check discipline as test_w1_z_comb_verify.py so the two
views cannot silently drift.
"""
import importlib.util
import math
import os
import types

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(modname, relpath):
    spec = importlib.util.spec_from_file_location(
        modname, os.path.join(_REPO, relpath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


zc = _load("_w1_z_comb_verify", "tests/_w1_z_comb_verify.py")


# ── relative_deviation + floor ───────────────────────────────────────────────
def test_relative_deviation_basic():
    # |1.0 - 0.5| / max(0.5, 0.02) = 0.5 / 0.5 = 1.0
    assert abs(zc.relative_deviation(1.0, 0.5, 0.02) - 1.0) < 1e-12


def test_relative_deviation_floor_guards_near_zero_ref():
    # ref ~ 0 -> denominator is the floor, not the tiny |ref|, so a small
    # absolute residual does not read as a huge relative one.
    assert abs(zc.relative_deviation(0.001, 0.0, 0.02) - 0.05) < 1e-12
    # without the floor this would be 0.001/epsilon -> blow up; the floor caps it.


# ── OLS slope + decay-rate fit ───────────────────────────────────────────────
def test_ols_slope_exact():
    assert abs(zc._ols_slope([0, 1, 2, 3], [0, 2, 4, 6]) - 2.0) < 1e-12
    assert zc._ols_slope([1], [1]) is None          # < 2 points
    assert zc._ols_slope([2, 2, 2], [1, 2, 3]) is None  # no x-variance


def test_fit_decay_rate_recovers_rate_on_even_kicks():
    # A(k) = exp(-0.03 k) on even kicks; odd kicks near zero (excluded).
    series = {}
    for k in range(0, 22):
        series[k] = math.exp(-0.03 * k) if k % 2 == 0 else 0.0005
    b = zc.fit_decay_rate(series, floor=0.02)
    assert b is not None and abs(b - (-0.03)) < 1e-9


def test_fit_decay_rate_too_few_points_returns_none():
    series = {0: 0.9, 2: 0.5}                        # only 2 even points
    assert zc.fit_decay_rate(series, floor=0.02) is None


# ── residual_report summary ──────────────────────────────────────────────────
def _synthetic(scale):
    """ref: even kicks 0.9 e^{-0.03k}, odd ~0; cand = ref * scale on the signal."""
    kicks = range(0, 60)
    ref_mean = {k: (0.9 * math.exp(-0.03 * k) if k % 2 == 0 else 0.0005) for k in kicks}
    ref_sem = {k: 0.0017 for k in kicks}
    cand = {k: ((ref_mean[k] * scale) if k % 2 == 0 else ref_mean[k], 0.0017)
            for k in kicks}
    return cand, ref_mean, ref_sem


def test_residual_report_flat_offset():
    cand, ref_mean, ref_sem = _synthetic(1.01)       # uniform +1%
    r = zc.residual_report(cand, ref_mean, ref_sem, floor=0.02)
    assert abs(r["max_rel_dev"] - 0.01) < 1e-6
    assert abs(r["mean_rel_dev"] - 0.01) < 1e-6
    assert r["mean_signed_residual"] > 0             # cand > ref
    assert abs(r["rel_dev_trend_slope"]) < 1e-6      # flat: no depth trend
    # identical decay shape -> matched rates
    assert r["decay_rate_rel_diff"] is not None
    assert r["decay_rate_rel_diff"] < 1e-6
    # only even kicks with |ref| >= floor are counted above-floor
    n_even_above = sum(1 for k in range(0, 60)
                       if k % 2 == 0 and 0.9 * math.exp(-0.03 * k) >= 0.02)
    assert r["n_above_floor"] == n_even_above


def test_residual_report_no_overlap_raises():
    with pytest.raises(ValueError, match="no overlapping"):
        zc.residual_report({999: (0.1, 0.01)}, {0: 0.9}, {0: 0.0017}, floor=0.02)


# ── integrated convergence verdict vs the pre-committed ceiling ──────────────
def _args(cand, floor=0.02, max_rel_dev=0.02, trend_warn=0.0):
    return types.SimpleNamespace(
        candidate="<synthetic>", reference="<synthetic>",
        candidate_seeds=40, reference_seeds=40, report=None,
        floor=floor, max_rel_dev=max_rel_dev, trend_warn=trend_warn,
    )


def test_verdict_converged_under_ceiling():
    cand, ref_mean, ref_sem = _synthetic(1.01)       # 1% < 2% ceiling
    assert zc._run_step1_residual(_args(cand), cand, ref_mean, ref_sem) == 0


def test_verdict_not_converged_over_ceiling():
    cand, ref_mean, ref_sem = _synthetic(1.05)       # 5% > 2% ceiling
    assert zc._run_step1_residual(_args(cand), cand, ref_mean, ref_sem) == 1


def test_ceiling_is_not_sigma_sys():
    # The ceiling (input) and the measured residual (output) are different
    # quantities: a 1% measured residual passes a 2% ceiling, and the
    # measured 1% — not the 2% — is what would parameterize the Option-1 gate.
    cand, ref_mean, ref_sem = _synthetic(1.01)
    r = zc.residual_report(cand, ref_mean, ref_sem, floor=0.02)
    assert r["max_rel_dev"] < 0.02                    # under ceiling
    assert abs(r["max_rel_dev"] - 0.01) < 1e-6        # sigma_sys = measured, ~1%
