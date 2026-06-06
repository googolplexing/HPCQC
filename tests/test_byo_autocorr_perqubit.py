# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Per-qubit autocorrelator un-collapse: parity invariant, self-describing
.dat, and the fail-loud width guard (RED-RULING-PER-QUBIT §1, §2.1, D1).

These cover the two PURE functions of the un-collapse, independent of the
engine wiring: the per-qubit observable and its self-describing aggregator.
The consumer-side ordering test (reading back through map_dtc_to_qpu_3d) and
the dedup-on equivalence gate live separately.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumi_hpc_qc.sweep.byo_observable import (
    aggregate_byo_autocorr_perqubit,
    get_autocorrelation,
    get_autocorrelation_perqubit,
)


def _random_counts(num_qubits, n_keys, rng):
    """Counts dict over length-num_qubits bitstrings (so the measured width
    equals num_qubits — the validated chain-measured path where the parity is
    exact)."""
    counts: dict[str, int] = {}
    for _ in range(n_keys):
        bits = "".join(str(b) for b in rng.integers(0, 2, num_qubits))
        counts[bits] = counts.get(bits, 0) + int(rng.integers(1, 25))
    return counts


def test_perqubit_mean_equals_legacy_scalar():
    """RED §1 parity invariant: mean over wires of the per-qubit vector
    reproduces the validated scalar (bit-for-bit when num_qub == num_qubits)."""
    rng = np.random.default_rng(0)
    nq = 6
    init = [0, 1, 0, 1, 1, 0]
    for _ in range(20):
        counts = _random_counts(nq, n_keys=40, rng=rng)
        scalar = get_autocorrelation(counts, init, nq)
        vec = get_autocorrelation_perqubit(counts, init, nq)
        assert vec.shape == (nq,)
        assert np.isclose(vec.mean(), scalar, atol=1e-12, rtol=0)


def test_perqubit_values_are_match_vs_init():
    """Deterministic per-wire values under the little-endian match-vs-init
    convention (same reversal as get_autocorrelation)."""
    # all-match -> every wire +1
    v = get_autocorrelation_perqubit({"000": 10}, [0, 0, 0], 3)
    assert np.allclose(v, [1.0, 1.0, 1.0])
    # bitstring "110" reverses to wires [0,1,1]; vs init [0,0,0] -> [+1,-1,-1]
    v = get_autocorrelation_perqubit({"110": 10}, [0, 0, 0], 3)
    assert np.allclose(v, [1.0, -1.0, -1.0])
    # mixed counts: 7x"000" (+1,+1,+1) and 3x"100"(rev[0,0,1]-> +1,+1,-1)
    v = get_autocorrelation_perqubit({"000": 7, "100": 3}, [0, 0, 0], 3)
    assert np.allclose(v, [1.0, 1.0, (7 - 3) / 10.0])


def test_perqubit_dat_is_self_describing_and_ordered(tmp_path):
    """RED §2.1: the physical qubit id is carried in the file and tracks the
    placement path order, even for a non-identity / non-monotonic set."""
    nk, nq = 4, 3
    phys = ["QB8", "QB16", "QB15"]  # deliberately non-monotonic (F5a HIGH-like)
    m0 = np.arange(nk * nq, dtype=float).reshape(nk, nq)
    m1 = m0 + 2.0
    mean, sem = aggregate_byo_autocorr_perqubit(
        [(0, m0), (1, m1)], phys, str(tmp_path)
    )
    assert mean.shape == (nk, nq) and sem.shape == (nk, nq)
    assert np.allclose(mean, (m0 + m1) / 2.0)

    path = tmp_path / "aggregated_autocorr_perqubit.dat"
    lines = path.read_text().splitlines()
    # RED D1 column names are preserved on the autocorr file even though the
    # writer is now the generic per-site helper underneath.
    header = lines[0]
    assert header.startswith("#")
    assert "kick" in header and "local_q" in header and "physical_q" in header
    rows = [ln.split() for ln in lines if not ln.startswith("#")]
    assert len(rows) == nk * nq
    for r in rows:
        kick, local_q, physical_q = int(r[0]), int(r[1]), r[2]
        # the file's own physical_q column must equal phys[local_q] — a consumer
        # reading the column cannot mis-map the site (the writer-level guarantee
        # the consumer ordering test then checks end-to-end through the reader).
        assert physical_q == phys[local_q]
        assert 0 <= kick < nk and 0 <= local_q < nq


def test_perqubit_aggregator_fails_loud_on_width_mismatch(tmp_path):
    """RED §2.1: never emit a .dat that silently mis-attributes sites."""
    m = np.zeros((4, 3))
    with pytest.raises(ValueError):
        aggregate_byo_autocorr_perqubit([(0, m)], ["QB1", "QB2"], str(tmp_path))
