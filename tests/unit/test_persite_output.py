# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Generic per-site time-series serializer (data.persite_output).

Tests the REUSABLE seam in isolation — observable-agnostic, so that any future
per-site observable (per-site polarization, classical-shadow per-site estimates)
can depend on it with confidence. The autocorr-specific behavior is tested
separately in test_byo_autocorr_perqubit.py.
"""
from __future__ import annotations

import numpy as np
import pytest

from lumi_hpc_qc.data.persite_output import write_persite_series


def test_self_describing_and_ordered(tmp_path):
    """The physical site id is carried in the file and tracks the caller's
    path order, even for a non-monotonic id set (RED §2.1)."""
    n_grid, n_site = 4, 3
    sites = ["QB8", "QB16", "QB15"]  # deliberately non-monotonic
    m0 = np.arange(n_grid * n_site, dtype=float).reshape(n_grid, n_site)
    m1 = m0 + 2.0
    mean, sem = write_persite_series(
        [(0, m0), (1, m1)], sites, str(tmp_path), filename="persite.dat"
    )
    assert mean.shape == (n_grid, n_site) and sem.shape == (n_grid, n_site)
    assert np.allclose(mean, (m0 + m1) / 2.0)

    rows = [
        ln.split()
        for ln in (tmp_path / "persite.dat").read_text().splitlines()
        if not ln.startswith("#")
    ]
    assert len(rows) == n_grid * n_site
    for r in rows:
        grid, local, physical = int(r[0]), int(r[1]), r[2]
        assert physical == sites[local]
        assert 0 <= grid < n_grid and 0 <= local < n_site


def test_single_instance_sem_is_zero(tmp_path):
    """A single instance has undefined ddof=1 sem; emit zeros, not nan, so the
    .dat stays numeric (matches aggregate_byo_autocorr)."""
    m = np.array([[0.5, -0.5]], dtype=float)
    mean, sem = write_persite_series(
        [(7, m)], ["QB1", "QB2"], str(tmp_path), filename="persite.dat"
    )
    assert np.allclose(mean, m)
    assert np.allclose(sem, 0.0)


def test_header_labels_are_configurable(tmp_path):
    """Header column names are caller-configurable (so the autocorr wrapper can
    pin RED-D1 names); the data rows are positional regardless."""
    m = np.zeros((2, 2))
    write_persite_series(
        [(0, m), (1, m)], ["A", "B"], str(tmp_path),
        filename="x.dat", grid_label="kick", local_label="local_q",
        physical_label="physical_q", value_label="autocorr",
    )
    header = (tmp_path / "x.dat").read_text().splitlines()[0]
    assert "kick" in header and "local_q" in header
    assert "physical_q" in header and "autocorr" in header


def test_fails_loud_on_width_mismatch(tmp_path):
    """Never emit a .dat that silently mis-attributes sites (RED §2.1)."""
    with pytest.raises(ValueError):
        write_persite_series(
            [(0, np.zeros((4, 3)))], ["QB1", "QB2"], str(tmp_path),
            filename="x.dat",
        )
