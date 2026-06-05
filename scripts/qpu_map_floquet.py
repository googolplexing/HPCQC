#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Paint Floquet-DTC / calibration data onto the physical Q50 lattice.

The Q50 layout (QB1..QB54) is transcribed from the device topology image and
verified against the calibration coupling map (every sampled edge lands on the
correct diagonal neighbor). QB32 has no per-qubit calibration entry (dead /
uncalibrated) but is a real lattice site, so it is drawn hollow.

Figures (into <out>/):
  qpu_quality_<metric>.png   device-quality landscape: each qubit colored by a
                             calibration metric (t2_us default), couplings drawn.
  qpu_coverage.png           if a results dir with byo_dat/ is given: each qubit
                             colored by how many of the placements used it
                             (shows the clustering of the solver's top-N).

Pure numpy + matplotlib + stdlib json. Run in the qiskit container:
  python qpu_map_floquet.py examples/q50_calibration_20260524_08c3c70f.json \\
      --results results/autocorr_echo_12p_10qb_10s_40k_1000s_19055840 \\
      --metric t2_us
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (column, band) per qubit; band 1 = top row of the device image, column 0 = left.
# Verified against calibration qubit_connectivity (diagonal nearest-neighbors).
LAYOUT = {
    "QB1": (9, 10), "QB2": (10, 9), "QB3": (6, 11), "QB4": (7, 10), "QB5": (8, 9),
    "QB6": (9, 8), "QB7": (10, 7), "QB8": (4, 11), "QB9": (5, 10), "QB10": (6, 9),
    "QB11": (7, 8), "QB12": (8, 7), "QB13": (9, 6), "QB14": (10, 5), "QB15": (2, 11),
    "QB16": (3, 10), "QB17": (4, 9), "QB18": (5, 8), "QB19": (6, 7), "QB20": (7, 6),
    "QB21": (8, 5), "QB22": (9, 4), "QB23": (1, 10), "QB24": (2, 9), "QB25": (3, 8),
    "QB26": (4, 7), "QB27": (5, 6), "QB28": (6, 5), "QB29": (7, 4), "QB30": (8, 3),
    "QB31": (9, 2), "QB32": (1, 8), "QB33": (2, 7), "QB34": (3, 6), "QB35": (4, 5),
    "QB36": (5, 4), "QB37": (6, 3), "QB38": (7, 2), "QB39": (8, 1), "QB40": (0, 7),
    "QB41": (1, 6), "QB42": (2, 5), "QB43": (3, 4), "QB44": (4, 3), "QB45": (5, 2),
    "QB46": (6, 1), "QB47": (0, 5), "QB48": (1, 4), "QB49": (2, 3), "QB50": (3, 2),
    "QB51": (4, 1), "QB52": (0, 3), "QB53": (1, 2), "QB54": (2, 1),
}


def xy(qb):
    """Screen coords: x = column, y = -band so band 1 is at the top (image-up)."""
    c, b = LAYOUT[qb]
    return c, -b


def load_cal(path):
    d = json.load(open(path))
    return d["qubits"], d.get("qubit_connectivity", [])


def discover_placements(results_dir):
    """Count how many placement dirs (byo_dat/.../{phys}/...) use each qubit."""
    counts = Counter()
    n_placements = 0
    if not results_dir:
        return counts, 0
    seen = set()
    for root, _dirs, files in os.walk(results_dir):
        if "aggregated_autocorr.dat" not in files:
            continue
        parts = os.path.normpath(root).split(os.sep)
        if len(parts) < 3:
            continue
        phys = parts[-3]  # .../{phys}/{env}/{obs}/aggregated_autocorr.dat
        if phys in seen:
            continue
        seen.add(phys)
        qbs = re.findall(r"QB\d+", phys)
        if not qbs:
            continue
        n_placements += 1
        for q in set(qbs):
            counts[q] += 1
    return counts, n_placements


def _draw_edges(ax, edges):
    for a, b in edges:
        if a in LAYOUT and b in LAYOUT:
            xa, ya = xy(a); xb, yb = xy(b)
            ax.plot([xa, xb], [ya, yb], color="0.6", lw=2.0, zorder=1)


def _draw_nodes(ax, values, *, cmap, vlabel, hollow=frozenset(), size=520):
    qbs = list(LAYOUT)
    vals = np.array([values.get(q, np.nan) for q in qbs], dtype=float)
    finite = vals[np.isfinite(vals)]
    vmin, vmax = (finite.min(), finite.max()) if finite.size else (0, 1)
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    for q in qbs:
        x, y = xy(q)
        v = values.get(q, np.nan)
        if q in hollow or not np.isfinite(v):
            ax.scatter([x], [y], s=size, marker="s", facecolors="none",
                       edgecolors="0.5", linewidths=1.5, zorder=3)
            ax.text(x, y, q.replace("QB", ""), ha="center", va="center",
                    fontsize=6, color="0.5", zorder=4)
        else:
            ax.scatter([x], [y], s=size, marker="s", c=[sm.to_rgba(v)],
                       edgecolors="k", linewidths=0.4, zorder=3)
            ax.text(x, y, q.replace("QB", ""), ha="center", va="center",
                    fontsize=6, color="k", zorder=4)
    return sm


def plot_quality(qubits, edges, metric, outdir):
    values = {q: qubits[q][metric] for q in qubits if metric in qubits[q]}
    uncal = {q for q in LAYOUT if q not in qubits}
    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_edges(ax, edges)
    cmap = "viridis_r" if "error" in metric else "viridis"
    sm = _draw_nodes(ax, values, cmap=cmap, vlabel=metric, hollow=uncal)
    cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cb.set_label(metric)
    ax.set_title(f"Q50 device — {metric}  (hollow = uncalibrated: "
                 f"{', '.join(sorted(uncal)) or 'none'})")
    ax.set_aspect("equal"); ax.axis("off")
    p = os.path.join(outdir, f"qpu_quality_{metric}.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def plot_coverage(counts, n_placements, edges, outdir):
    fig, ax = plt.subplots(figsize=(9, 9))
    _draw_edges(ax, edges)
    values = {q: float(counts.get(q, 0)) for q in LAYOUT}
    sm = _draw_nodes(ax, {q: v for q, v in values.items() if v > 0},
                     cmap="inferno", vlabel="placements")
    # qubits used by no placement: draw faint
    for q in LAYOUT:
        if values[q] == 0:
            x, y = xy(q)
            ax.scatter([x], [y], s=520, marker="s", facecolors="0.93",
                       edgecolors="0.8", linewidths=0.5, zorder=2)
            ax.text(x, y, q.replace("QB", ""), ha="center", va="center",
                    fontsize=6, color="0.6", zorder=4)
    if sm is not None:
        cb = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
        cb.set_label("# placements using this qubit")
    ax.set_title(f"Placement coverage across {n_placements} placement(s)")
    ax.set_aspect("equal"); ax.axis("off")
    p = os.path.join(outdir, "qpu_coverage.png")
    fig.tight_layout(); fig.savefig(p, dpi=150); plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calibration", help="q50 calibration JSON")
    ap.add_argument("--results", default=None, help="results dir with byo_dat/")
    ap.add_argument("--metric", default="t2_us",
                    choices=["t1_us", "t2_us", "t2_echo_us",
                             "readout_fidelity", "single_gate_error"])
    ap.add_argument("--out", default="qpu_figures")
    args = ap.parse_args(argv)

    qubits, edges = load_cal(args.calibration)
    os.makedirs(args.out, exist_ok=True)
    written = [plot_quality(qubits, edges, args.metric, args.out)]

    counts, n_pl = discover_placements(args.results)
    if n_pl:
        written.append(plot_coverage(counts, n_pl, edges, args.out))

    print(f"{len(LAYOUT)} sites, {len(qubits)} calibrated, "
          f"{len(LAYOUT) - len(qubits)} uncalibrated; {n_pl} placement(s)")
    for p in written:
        print("  wrote", p)


if __name__ == "__main__":
    main()
