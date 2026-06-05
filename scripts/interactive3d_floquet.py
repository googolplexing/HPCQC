#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Interactive, fully-rotatable 3D views of the Q50 / Floquet-DTC data (plotly).

Emits self-contained HTML files (plotly.js inlined, so they open offline in any
browser and rotate / zoom / pan around all three axes with the mouse):

  qpu_lattice_3d.html     the Q50 lattice as a quality terrain -- qubits at their
                          verified (x, y) lattice positions, height z = chosen
                          calibration metric, couplings as 3D bonds, hover shows
                          all per-qubit values. QB32 (uncalibrated) sits at z=0.
  results_surface_3d.html (if --results) a rotatable surface of the actual data:
                          x = Floquet kick, y = placement, z = mean autocorrelator.

The data-prep (coordinates, byo_dat parsing) is shared with qpu_map_floquet.py and
visualize_floquet_dtc.py. Run in the qiskit container (plotly 6.6.0 present):

  python interactive3d_floquet.py examples/q50_calibration_20260524_08c3c70f.json \\
      --results results/autocorr_echo_12p_10qb_10s_40k_1000s_19055840 \\
      --metric t2_echo_us
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

# (column, band); band 1 = top of the device image, column 0 = left. Verified
# against the calibration coupling map (diagonal nearest-neighbors). Same table
# as qpu_map_floquet.py.
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
METRICS = ["t1_us", "t2_us", "t2_echo_us", "readout_fidelity", "single_gate_error"]


def xy(qb):
    c, b = LAYOUT[qb]
    return c, -b  # band 1 at the top


def load_cal(path):
    d = json.load(open(path))
    return d["qubits"], d.get("qubit_connectivity", [])


def lattice_arrays(qubits, metric):
    """Node arrays (x, y, z, color, text) and bond segment arrays for plotly."""
    nx, ny, nz, nc, nt = [], [], [], [], []
    vals = [qubits[q][metric] for q in qubits if metric in qubits[q]]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    span = (vmax - vmin) or 1.0
    for q in LAYOUT:
        x, y = xy(q)
        nx.append(x); ny.append(y)
        if q in qubits and metric in qubits[q]:
            v = qubits[q][metric]
            nz.append((v - vmin) / span)  # normalized height
            nc.append(v)
            props = qubits[q]
            nt.append(f"{q}<br>" + "<br>".join(
                f"{m}={props[m]:.3g}" for m in METRICS if m in props))
        else:
            nz.append(0.0); nc.append(None)
            nt.append(f"{q}<br>(uncalibrated)")
    return dict(x=nx, y=ny, z=nz, color=nc, text=nt,
                vmin=vmin, vmax=vmax, znorm=lambda v: (v - vmin) / span)


def bond_segments(edges, znorm_of_qb):
    """Edge line segments with None separators (plotly multi-segment trick)."""
    ex, ey, ez = [], [], []
    for a, b in edges:
        if a in LAYOUT and b in LAYOUT:
            xa, ya = xy(a); xb, yb = xy(b)
            ex += [xa, xb, None]; ey += [ya, yb, None]
            ez += [znorm_of_qb(a), znorm_of_qb(b), None]
    return ex, ey, ez


def results_matrix(results_dir):
    """(placements, kicks, M[placement, kick]) of mean autocorr for the autocorr arm."""
    rows = {}
    for root, _dirs, files in os.walk(results_dir):
        if "aggregated_autocorr.dat" not in files:
            continue
        parts = os.path.normpath(root).split(os.sep)
        if len(parts) < 3:
            continue
        obs, _env, phys = parts[-1], parts[-2], parts[-3]
        if obs not in ("autocorr",):  # surface from the autocorr arm
            continue
        arr = np.loadtxt(os.path.join(root, "aggregated_autocorr.dat"), comments="#")
        if arr.ndim == 1:
            arr = arr[None, :]
        rows[phys] = arr  # cols: kick, mean, sem
    if not rows:
        return [], np.array([]), np.array([])
    placements = sorted(rows)
    kicks = rows[placements[0]][:, 0]
    M = np.array([rows[p][:, 1] for p in placements])
    return placements, kicks, M


def write_lattice_html(qubits, edges, metric, path):
    import plotly.graph_objects as go
    a = lattice_arrays(qubits, metric)
    znorm_of = lambda q: a["znorm"](qubits[q][metric]) if (
        q in qubits and metric in qubits[q]) else 0.0
    ex, ey, ez = bond_segments(edges, znorm_of)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color="gray", width=3),
                               hoverinfo="skip", name="couplings"))
    cal_idx = [i for i, c in enumerate(a["color"]) if c is not None]
    unc_idx = [i for i, c in enumerate(a["color"]) if c is None]
    fig.add_trace(go.Scatter3d(
        x=[a["x"][i] for i in cal_idx], y=[a["y"][i] for i in cal_idx],
        z=[a["z"][i] for i in cal_idx], mode="markers+text",
        marker=dict(size=7, color=[a["color"][i] for i in cal_idx],
                    colorscale="Viridis", colorbar=dict(title=metric),
                    cmin=a["vmin"], cmax=a["vmax"], line=dict(width=0.5, color="black")),
        text=[t.split("<br>")[0].replace("QB", "") for t in
              [a["text"][i] for i in cal_idx]],
        textposition="top center", textfont=dict(size=8),
        hovertext=[a["text"][i] for i in cal_idx], hoverinfo="text", name=metric))
    if unc_idx:
        fig.add_trace(go.Scatter3d(
            x=[a["x"][i] for i in unc_idx], y=[a["y"][i] for i in unc_idx],
            z=[a["z"][i] for i in unc_idx], mode="markers+text",
            marker=dict(size=7, color="lightgray", symbol="x"),
            text=[a["text"][i].split("<br>")[0].replace("QB", "") for i in unc_idx],
            hovertext=[a["text"][i] for i in unc_idx], hoverinfo="text",
            name="uncalibrated"))
    fig.update_layout(
        title=f"Q50 lattice terrain — height & color = {metric}",
        scene=dict(xaxis_title="lattice column", yaxis_title="lattice band",
                   zaxis_title=f"{metric} (normalized)",
                   aspectmode="manual", aspectratio=dict(x=1.4, y=1.4, z=0.7)),
        showlegend=True)
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def write_surface_html(results_dir, path):
    import plotly.graph_objects as go
    placements, kicks, M = results_matrix(results_dir)
    if not placements:
        return None
    fig = go.Figure(data=[go.Surface(
        z=M, x=kicks, y=np.arange(len(placements)),
        colorscale="RdBu", cmid=0, colorbar=dict(title="autocorr"))])
    fig.update_layout(
        title="Mean autocorrelator surface — rotate to see the period-2 ridges",
        scene=dict(xaxis_title="Floquet kick", yaxis_title="placement index",
                   zaxis_title="autocorrelator",
                   yaxis=dict(tickmode="array", tickvals=list(range(len(placements))),
                              ticktext=placements)))
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calibration")
    ap.add_argument("--results", default=None)
    ap.add_argument("--metric", default="t2_echo_us", choices=METRICS)
    ap.add_argument("--out", default="qpu_3d")
    args = ap.parse_args(argv)

    qubits, edges = load_cal(args.calibration)
    os.makedirs(args.out, exist_ok=True)
    written = [write_lattice_html(qubits, edges, args.metric,
                                  os.path.join(args.out, "qpu_lattice_3d.html"))]
    if args.results:
        s = write_surface_html(args.results, os.path.join(args.out, "results_surface_3d.html"))
        if s:
            written.append(s)
    for p in written:
        print("wrote", p)


if __name__ == "__main__":
    main()
