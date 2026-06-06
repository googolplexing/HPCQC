# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Generic per-site (site-resolved) time-series serialization.

Observable-agnostic output helper shared across simulation/observable types:
the BYO autocorrelator un-collapse uses it today, and any other per-site
time-series observable (per-site polarization, classical-shadow per-site
estimates, etc.) reuses it rather than re-implementing the aggregation and the
self-describing .dat format. It carries the physical site id IN the file so a
consumer cannot mis-map a site via a separate lookup
(RED-RULING-PER-QUBIT-AUTOCORRELATOR-AND-SITE-RESOLVED-RUN-v1.0 §2.1, D1).

numpy-only (no qiskit / h5py), so it is importable from any layer — the live
engine aggregation path and the merge-time reducers alike.
"""
from __future__ import annotations

import os

import numpy as np


def write_persite_series(
    per_seed_matrices,
    site_ids,
    out_dir,
    *,
    filename,
    grid_label="grid",
    local_label="local_site",
    physical_label="physical_site",
    value_label="value",
):
    """Average per-instance per-site arrays over the instance axis and write a
    SELF-DESCRIBING per-site .dat.

    mean/sem are computed independently per ``(grid_point, site)`` over the
    instance (seed) axis, so the site axis is preserved automatically — the same
    aggregation generalizes to any per-site observable; only the value/column
    naming differs. The physical site id is written at column ``physical_*`` so
    the attribution travels with the data (RED §2.1). ``local -> physical`` is
    the caller's path order.

    Args:
      per_seed_matrices: list of ``(instance_key, matrix)``; each matrix has
                         shape ``(N_grid, N_site)`` — one row per grid point
                         (e.g. Floquet kick), one column per site.
      site_ids: ordered local->physical site ids, length N_site (path order).
      out_dir: directory to write into (created if absent).
      filename: output .dat filename.
      grid_label/local_label/physical_label/value_label: header column names
        (the data ROWS are positional, so these are human-facing only — a
        consumer reads by column position, not header text).

    Writes ``<out_dir>/<filename>``:
      "# {grid}  {local}  {physical}  mean_{value}  sem"
      "{grid:4d} {local:4d} {physical:>10} {mean:10.4f} {sem:10.4f}"
      sem = std(ddof=1)/sqrt(N) per (grid, site); zeros for a single instance.

    Returns: (mean, sem) numpy arrays of shape (N_grid, N_site).
    """
    os.makedirs(out_dir, exist_ok=True)
    stack = np.array(
        [np.asarray(m, dtype=float) for _, m in per_seed_matrices]
    )  # (N_inst, N_grid, N_site)
    n_inst, n_grid, n_site = stack.shape
    sites = [str(s) for s in site_ids]
    # Fail loud on a local->physical length mismatch rather than emit a .dat that
    # silently mis-attributes sites (RED §2.1: attribution is load-bearing).
    if len(sites) != n_site:
        raise ValueError(
            f"per-site width {n_site} != len(site_ids) {len(sites)}"
        )
    mean = stack.mean(axis=0)
    if n_inst > 1:
        sem = stack.std(axis=0, ddof=1) / np.sqrt(n_inst)
    else:
        sem = np.zeros((n_grid, n_site))

    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            f"# {grid_label}  {local_label}  {physical_label}  "
            f"mean_{value_label}  sem\n"
        )
        for g in range(n_grid):
            for s in range(n_site):
                f.write(
                    f"{g:4d} {s:4d} {sites[s]:>10} "
                    f"{mean[g, s]:10.4f} {sem[g, s]:10.4f}\n"
                )

    return mean, sem
