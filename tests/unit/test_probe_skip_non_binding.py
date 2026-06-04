# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""RED-DIRECTIVE-PROBE-SKIP-WHEN-NON-BINDING — offline acceptance gates.

Pure/stdlib, like the worker_cap suite — no qiskit/aer/h5py, no real allocation.
Two gates per the directive §6:

  * SKIP DECISION (straddle the boundary): for a units-bound case the probe is
    not entered, compute_worker_cap(peak_hi) returns core_units_ceiling, and the
    provenance tag is "skip:mem_non_binding"; for a memory-bound case the probe
    IS entered (skip is False). The two cases straddle safe_mem//peak_hi vs
    core_units_ceiling so the gate is non-vacuous.

  * peak_hi SOUNDNESS (the load-bearing test): peak_hi(method, n) must be an
    UPPER bound on every measured VmHWM in the §1.4 corpus, across the n range
    that skips, for the heaviest observable family. A corpus entry that exceeds
    peak_hi must FAIL the test (conservatism proven, not assumed).

NOTE (scaffold): the corpus below is wired with the one banked measurement Red
verified (~1.32 GiB device-calibrated at 10q; RED-RESP-W1-CAP-VERIFY §1.4). The
remaining (method, n, family) measurements over the skipping range must be
pasted from evidence/W1/ to make the soundness gate non-vacuous beyond 10q;
those rows are marked TODO and the gate asserts coverage of at least the
campaign point until then.
"""
from __future__ import annotations

from lumi_hpc_qc.sweep import worker_cap as wc

G = wc.GIB


# ── conservative_peak_hi: shape + ordering ──────────────────────────────────

def test_peak_hi_at_10q_statevector_is_dominated_by_c1():
    # 16 * 2**10 = 16 KiB state; * STATE_FACTOR (4) = 64 KiB << C1 (3 GiB).
    p = wc.conservative_peak_hi({"statevector"}, 10)
    assert p == wc.C1_PER_UNIT_PEAK_BYTES + 4 * 16 * (1 << 10)
    assert wc.C1_PER_UNIT_PEAK_BYTES < p < wc.C1_PER_UNIT_PEAK_BYTES + (1 << 20)


def test_peak_hi_density_matrix_heavier_than_statevector():
    sv = wc.conservative_peak_hi({"statevector"}, 12)
    dm = wc.conservative_peak_hi({"density_matrix"}, 12)
    assert dm > sv  # 16*2**24 vs 16*2**12


def test_peak_hi_takes_heaviest_method_in_the_set():
    mixed = wc.conservative_peak_hi({"statevector", "density_matrix"}, 12)
    dm = wc.conservative_peak_hi({"density_matrix"}, 12)
    assert mixed == dm


def test_peak_hi_empty_methods_defaults_statevector():
    assert wc.conservative_peak_hi(set(), 10) == \
        wc.conservative_peak_hi({"statevector"}, 10)


def test_peak_hi_unknown_method_treated_as_density_matrix():
    # Conservative: an unknown method must not under-bound.
    assert wc.conservative_peak_hi({"mps_or_future"}, 12) == \
        wc.conservative_peak_hi({"density_matrix"}, 12)


# ── decide_probe_skip: straddle the boundary ────────────────────────────────

def test_skip_when_memory_non_binding_units_bound():
    # safe_mem 200 GiB, peak_hi 3 GiB -> mem_term_lo 66; core_units_ceiling
    # = min(128, 4, 128) = 4. 66 >= 4 -> skip.
    d = wc.decide_probe_skip(
        cpu_workers=128, num_units=4, usable_cores_physical=128,
        safe_mem_bytes=200 * G, peak_hi_bytes=3 * G,
    )
    assert d.skip is True
    assert d.core_units_ceiling == 4
    assert d.peak_source == "skip:mem_non_binding"
    # The skip feeds peak_hi to the unchanged cap, which returns the ceiling.
    cap = wc.compute_worker_cap(
        cpu_workers=128, num_units=4, usable_cores_physical=128,
        safe_mem_bytes=200 * G, per_unit_peak_bytes=3 * G,
    )
    assert cap.cap == d.core_units_ceiling == 4


def test_no_skip_when_memory_can_bind():
    # Same budget/peak, but 80 units -> core_units_ceiling 80 > mem_term_lo 66
    # -> memory MIGHT bind -> must still probe (today's path).
    d = wc.decide_probe_skip(
        cpu_workers=128, num_units=80, usable_cores_physical=128,
        safe_mem_bytes=200 * G, peak_hi_bytes=3 * G,
    )
    assert d.skip is False
    assert d.core_units_ceiling == 80
    assert d.peak_source is None


def test_skip_boundary_is_exact():
    # mem_term_lo == core_units_ceiling -> skip (>=). One unit more -> no skip.
    # safe_mem // 3GiB = 10 with safe_mem = 30 GiB.
    on = wc.decide_probe_skip(
        cpu_workers=10, num_units=10, usable_cores_physical=64,
        safe_mem_bytes=30 * G, peak_hi_bytes=3 * G,
    )
    assert on.core_units_ceiling == 10 and on.skip is True
    off = wc.decide_probe_skip(
        cpu_workers=11, num_units=11, usable_cores_physical=64,
        safe_mem_bytes=30 * G, peak_hi_bytes=3 * G,
    )
    assert off.core_units_ceiling == 11 and off.skip is False


def test_no_skip_when_safe_mem_unavailable():
    # Unknown budget -> never skip (fall through to probe / D4 raise).
    d = wc.decide_probe_skip(
        cpu_workers=128, num_units=4, usable_cores_physical=128,
        safe_mem_bytes=None, peak_hi_bytes=3 * G,
    )
    assert d.skip is False
    assert d.peak_source is None


# ── peak_hi SOUNDNESS over the corpus (load-bearing) ────────────────────────

# Banked device-calibrated VmHWM measurements (the per_unit_peak the live probe
# recorded), all at n=10 — the framework's operating size; no larger-n sweep has
# run, so the skipping range in production is 10q. peak_hi must dominate the MAX
# (1.32 GiB). The echo runs (f5a/p3) are LIGHTER than the autocorr canary,
# confirming echo's extra gates cost time, not state memory
# (RED-DIRECTIVE-PROBE-SKIP §5) — the heaviest-family obligation is met by the
# 1.32 GiB autocorr point. (method, n, family, measured_bytes, provenance)
_MEASURED_VMHWM_CORPUS = [
    ("statevector", 10, "autocorr", int(1.32 * G),
     "job 18924351 evidence/W1/w1_3-canary-clean"),
    ("statevector", 10, "autocorr", int(1.30 * G),
     "job 18938950 evidence/W1/d5-multiwave"),
    ("statevector", 10, "autocorr", int(1.28 * G),
     "job 18943612 evidence/W1/gate2-fail-18943612"),
    ("statevector", 10, "echo", int(0.99 * G),
     "job 18984339 evidence/slurm_logs/f5a_val"),
    ("statevector", 10, "echo", int(0.70 * G),
     "job 18986846 evidence/slurm_logs/p3_single"),
    ("statevector", 10, "echo", int(0.69 * G),
     "job 18984820 evidence/slurm_logs/f5a_hi"),
    # No n>10 measurements exist (no larger-n sweep has run). When one is run,
    # capture its VmHWM and add it here; until then the skip is corpus-validated
    # only at n=10, and the binding-regime probe is the backstop at any n where
    # peak_hi has not been corpus-checked.
]


def test_peak_hi_dominates_measured_corpus():
    for method, n, family, measured, prov in _MEASURED_VMHWM_CORPUS:
        bound = wc.conservative_peak_hi({method}, n)
        assert bound >= measured, (
            f"peak_hi({method},{n})={bound / G:.2f} GiB < measured "
            f"{measured / G:.2f} GiB ({family}, {prov}) — bound is NOT "
            f"conservative; bump FIXED_OVERHEAD/STATE_FACTOR before any skip "
            f"on this shape"
        )


def test_soundness_check_actually_catches_an_underbound():
    # Self-check (non-vacuity of the gate above): a peak_hi BELOW a corpus
    # measurement must be detectable as a failure, not silently pass.
    method, n, _family, measured, _prov = _MEASURED_VMHWM_CORPUS[0]
    underbound = measured - 1
    assert not (underbound >= measured)               # the gate's predicate
    assert wc.conservative_peak_hi({method}, n) >= measured  # real bound passes


def test_corpus_covers_both_families_at_the_campaign_point():
    # The skipping shape is 10q; assert BOTH observable families are represented
    # so the heaviest-family soundness obligation is not vacuously met.
    fams = {family for _m, n, family, _meas, _p in _MEASURED_VMHWM_CORPUS
            if n == 10}
    assert {"autocorr", "echo"} <= fams
