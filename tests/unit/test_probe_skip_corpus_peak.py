# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Corpus-measured probe-skip bound — offline acceptance gate.

BLUE-TO-RED-CORPUS-PEAK-PROBE-SKIP: on a corpus-validated shape the probe-skip
sizes on the MEASURED per-unit peak (CORPUS_MEASURED_PEAK_BYTES) instead of the
conservative C1 bound, so a provably non-binding run skips the probe AT FULL
concurrency. Pure/stdlib, like the rest of the worker_cap suite.

The load-bearing test is `test_corpus_peak_flips_the_campaign_case`: the exact
regime that today runs the probe (conservative bound -> memory looks binding)
must SKIP under the corpus bound — that single assertion is the whole point of
the change and proves it is non-vacuous (it flips, the conservative case does
not).
"""
from __future__ import annotations

from lumi_hpc_qc.sweep import worker_cap as wc

G = wc.GIB


# ── keys must never drift from the validation set ───────────────────────────

def test_measured_keys_match_validated_shapes():
    # A shape can be skip-eligible (CORPUS_VALIDATED_PEAK_HI_SHAPES) iff it has a
    # measured peak to size the cap on — the two sets must stay identical, or a
    # shape could skip with no measured bound (conservative) or carry a measured
    # bound it is never allowed to use.
    assert set(wc.CORPUS_MEASURED_PEAK_BYTES) == set(
        wc.CORPUS_VALIDATED_PEAK_HI_SHAPES
    )


# ── corpus_measured_peak_hi: value, every-method semantics, None fallback ────

def test_corpus_peak_is_measured_value_times_factor():
    got = wc.corpus_measured_peak_hi({"statevector"}, 10)
    assert got == int(1.32 * G * wc.CORPUS_PEAK_SAFETY_FACTOR)


def test_corpus_peak_none_for_unmeasured_shape():
    assert wc.corpus_measured_peak_hi({"statevector"}, 12) is None
    assert wc.corpus_measured_peak_hi({"density_matrix"}, 10) is None


def test_corpus_peak_requires_every_method_measured():
    # statevector@10 is measured, density_matrix@10 is not -> mixed group has no
    # corpus bound (cannot under-size on a half-measured group).
    assert wc.corpus_measured_peak_hi({"statevector", "density_matrix"}, 10) is None


def test_corpus_peak_empty_methods_defaults_statevector():
    assert wc.corpus_measured_peak_hi(set(), 10) == \
        wc.corpus_measured_peak_hi({"statevector"}, 10)


# ── soundness: the corpus bound is a valid, tighter upper bound ─────────────

def test_corpus_bound_between_measured_and_conservative():
    measured = wc.CORPUS_MEASURED_PEAK_BYTES[("statevector", 10)]
    corpus = wc.corpus_measured_peak_hi({"statevector"}, 10)
    conservative = wc.conservative_peak_hi({"statevector"}, 10)
    # >= measured true peak (OOM-safe); <= conservative (it is the tighter bound
    # the whole change rests on).
    assert measured <= corpus <= conservative
    assert wc.CORPUS_PEAK_SAFETY_FACTOR >= 1.0


# ── the load-bearing flip: campaign regime probes conservative, skips corpus ─

def test_corpus_peak_flips_the_campaign_case():
    # The banked campaign on this rank: 120 device-cal statevector@10 units,
    # 128 cores, ~197 GiB safe_mem. core_units_ceiling = min(128, 120, 128) = 120.
    safe_mem = 197 * G
    methods, n = {"statevector"}, 10

    conservative = wc.conservative_peak_hi(methods, n)            # ~3 GiB
    corpus = wc.corpus_measured_peak_hi(methods, n)               # ~1.32 GiB
    assert corpus is not None

    d_conservative = wc.decide_probe_skip(
        cpu_workers=128, num_units=120, usable_cores_physical=128,
        safe_mem_bytes=safe_mem, peak_hi_bytes=conservative,
    )
    d_corpus = wc.decide_probe_skip(
        cpu_workers=128, num_units=120, usable_cores_physical=128,
        safe_mem_bytes=safe_mem, peak_hi_bytes=corpus,
    )

    # Conservative bound: memory looks binding (197 // 3 = 65 < 120) -> PROBE.
    assert d_conservative.skip is False
    # Corpus bound: memory provably non-binding (197 // 1.32 = 149 >= 120) -> SKIP.
    assert d_corpus.skip is True
    assert d_corpus.core_units_ceiling == 120


def test_skipped_cap_is_full_concurrency_under_corpus_bound():
    # When the corpus-bound skip fires, compute_worker_cap(corpus_peak) must
    # return the full core_units_ceiling (no under-pack) and not raise D4.
    corpus = wc.corpus_measured_peak_hi({"statevector"}, 10)
    decision = wc.compute_worker_cap(
        cpu_workers=128, num_units=120, usable_cores_physical=128,
        safe_mem_bytes=197 * G, per_unit_peak_bytes=corpus,
    )
    assert decision.cap == 120


def test_corpus_bound_still_oom_safe_packing():
    # The skip OOM proof: core_units_ceiling * corpus_bound <= safe_mem.
    corpus = wc.corpus_measured_peak_hi({"statevector"}, 10)
    safe_mem = 197 * G
    d = wc.decide_probe_skip(
        cpu_workers=128, num_units=120, usable_cores_physical=128,
        safe_mem_bytes=safe_mem, peak_hi_bytes=corpus,
    )
    assert d.skip is True
    assert d.core_units_ceiling * corpus <= safe_mem
