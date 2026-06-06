#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Stacked-sheet spatiotemporal view of a Floquet-DTC sweep on the Q50 lattice.

Unlike map_dtc_to_qpu_3d.py (which collapses each qubit's whole A(kick) trace to
a single subharmonic scalar -> one flat terrain), this keeps the Floquet kick as
its own axis. For every kick it draws ONE translucent sheet at height z = kick,
with the sheet's vertices sitting at the sampled qubits' physical lattice
positions and colored by that qubit's autocorrelator at that kick. Stacked up the
z-axis, the sheets are the 3D version of the placement x kick heatmap, wrapped
onto the chip geometry: colors flip red<->blue every kick (the period-2 DTC
signal) and fade to white as a site decoheres.

Reads the per-qubit .dat (kick local_q physical_q mean sem) emitted by the P3
reducer. A qubit touched by several chains is the mean over those chains
(overlap-average), matching map_dtc_to_qpu_3d.py.

FAITHFULNESS NOTE: each sheet is a Delaunay triangulation (in the xy-plane) of the
*sampled* qubits only, so it spans their convex hull and shades linearly between
real measurements -- it does NOT extrapolate onto unsampled chip regions, but it
does smooth across any uncovered sites that fall inside the hull. Vertices are
always real data.

Emits (into <out>/):
  qpu_dtc_volume_3d.html   rotatable stack of per-kick sheets over the Q50 lattice.

Pure numpy + plotly (lazy import) + stdlib json. Run in the qiskit container:
  python qpu_dtc_volume_3d.py examples/q50_calibration_20260606_51b18c0c.json \
      results/smoke4_2node_job19083859 --arm autocorr --out results/.../volume
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np

# (column, band); band 1 = top of the device image, column 0 = left. Verified
# against the calibration coupling map (diagonal nearest-neighbors). Same table
# as map_dtc_to_qpu_3d.py / qpu_map_floquet.py.
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
    return d.get("qubits", {}), d.get("qubit_connectivity", [])


def read_perqubit(results_dir, arm):
    """Walk results_dir; aggregate per-qubit A(kick) over chains for the given arm.

    Returns (kicks, series, coverage, n_leaves):
      kicks    : sorted list of kick indices
      series   : {physical_q: np.ndarray aligned to kicks (NaN where missing)}
      coverage : {physical_q: number of chains that included this qubit}
      n_leaves : number of per-qubit .dat leaves consumed
    """
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))  # phys -> kick -> [sum, count]
    cover = defaultdict(set)                                   # phys -> {placement dir}
    kicks_seen = set()
    n_leaves = 0
    for root, _dirs, files in os.walk(results_dir):
        if "aggregated_autocorr_perqubit.dat" not in files:
            continue
        parts = os.path.normpath(root).split(os.sep)
        if len(parts) < 3 or parts[-1] != arm:
            continue
        placement = parts[-3]
        n_leaves += 1
        with open(os.path.join(root, "aggregated_autocorr_perqubit.dat")) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split()
                if len(p) < 4:
                    continue
                kick = int(float(p[0]))
                pq = p[2]
                mean = float(p[3])
                acc[pq][kick][0] += mean
                acc[pq][kick][1] += 1
                cover[pq].add(placement)
                kicks_seen.add(kick)
    kicks = sorted(kicks_seen)
    series, coverage = {}, {}
    for pq, kd in acc.items():
        series[pq] = np.array(
            [(kd[k][0] / kd[k][1]) if (k in kd and kd[k][1]) else np.nan for k in kicks]
        )
        coverage[pq] = len(cover[pq])
    return kicks, series, coverage, n_leaves


def _edges_trace(edges):
    import plotly.graph_objects as go
    ex, ey, ez = [], [], []
    for a, b in edges:
        if a in LAYOUT and b in LAYOUT:
            xa, ya = xy(a)
            xb, yb = xy(b)
            ex += [xa, xb, None]
            ey += [ya, yb, None]
            ez += [0, 0, None]
    return go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                        line=dict(color="lightgray", width=2),
                        hoverinfo="skip", name="Q50 couplings")


def write_volume_html(kicks, series, coverage, edges, path, *, arm, stagger,
                      stride, opacity, guides):
    import plotly.graph_objects as go

    lit = [q for q in series if q in LAYOUT]
    unmatched = [q for q in series if q not in LAYOUT]
    if len(lit) < 3:
        raise SystemExit(f"only {len(lit)} sampled qubit(s) map to the lattice; "
                         "need >=3 to triangulate a sheet")

    karr = np.array(kicks, dtype=float)
    xs = np.array([xy(q)[0] for q in lit], dtype=float)
    ys = np.array([xy(q)[1] for q in lit], dtype=float)

    # value matrix: rows = lit qubits, cols = kicks (NaN where missing)
    M = np.vstack([series[q] for q in lit])  # (nq, nk)
    if stagger:
        M = ((-1.0) ** karr)[None, :] * M
    vmax = float(np.nanmax(np.abs(M))) or 1.0
    clabel = "staggered A" if stagger else f"{arm} A"

    fig = go.Figure()
    fig.add_trace(_edges_trace(edges))

    # faint base markers for every site; lit ones carry hover (qubit + coverage)
    fig.add_trace(go.Scatter3d(
        x=[xy(q)[0] for q in LAYOUT], y=[xy(q)[1] for q in LAYOUT],
        z=[0.0] * len(LAYOUT), mode="markers",
        marker=dict(size=2, color="lightgray"), hoverinfo="skip", name="sites"))
    fig.add_trace(go.Scatter3d(
        x=xs, y=ys, z=[0.0] * len(lit), mode="markers",
        marker=dict(size=4, color="black"),
        hovertext=[f"{q}<br>{coverage[q]} chain(s)" for q in lit],
        hoverinfo="text", name="sampled qubits"))

    # optional vertical guide lines per lit qubit (helps read each column)
    if guides:
        gx, gy, gz = [], [], []
        for x0, y0 in zip(xs, ys):
            gx += [x0, x0, None]
            gy += [y0, y0, None]
            gz += [0.0, float(karr.max()), None]
        fig.add_trace(go.Scatter3d(x=gx, y=gy, z=gz, mode="lines",
                                   line=dict(color="rgba(150,150,150,0.25)", width=1),
                                   hoverinfo="skip", name="qubit columns"))

    # one translucent sheet per kick
    first = True
    n_sheets = 0
    for j, k in enumerate(kicks):
        if j % stride:
            continue
        vals = M[:, j]
        fin = np.isfinite(vals)
        if int(fin.sum()) < 3:
            continue
        fig.add_trace(go.Mesh3d(
            x=xs[fin], y=ys[fin], z=np.full(int(fin.sum()), float(k)),
            intensity=vals[fin], intensitymode="vertex",
            colorscale="RdBu", reversescale=True, cmin=-vmax, cmax=vmax,
            delaunayaxis="z", opacity=opacity, flatshading=False,
            showscale=first,
            colorbar=dict(title=clabel) if first else None,
            name=f"kick {k}", hoverinfo="skip"))
        first = False
        n_sheets += 1

    fig.update_layout(
        title=(f"Q50 spatiotemporal DTC volume - {arm} arm"
               + (" (staggered)" if stagger else "")
               + f"  |  {len(lit)} sampled qubits, {n_sheets} kick-sheets"),
        scene=dict(xaxis_title="lattice column", yaxis_title="lattice band",
                   zaxis_title="Floquet kick",
                   aspectmode="manual", aspectratio=dict(x=1.2, y=1.2, z=1.5)),
        showlegend=True)
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path, lit, unmatched, n_sheets, vmax


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("calibration", help="q50 calibration JSON (for the coupling base layer)")
    ap.add_argument("results", help="merged results dir with byo_dat/")
    ap.add_argument("--arm", default="autocorr", choices=["autocorr", "echo"],
                    help="which observable arm's per-qubit data to render")
    ap.add_argument("--stagger", action="store_true",
                    help="color by (-1)^kick * A (decay envelope) instead of raw A")
    ap.add_argument("--kick-stride", type=int, default=1, dest="stride",
                    help="draw every Nth kick-sheet (thin a dense/slow stack)")
    ap.add_argument("--opacity", type=float, default=0.30,
                    help="per-sheet opacity (lower = see deeper into the stack)")
    ap.add_argument("--no-guides", action="store_false", dest="guides",
                    help="omit the faint vertical per-qubit guide lines")
    ap.add_argument("--out", default="qpu_dtc_volume")
    args = ap.parse_args(argv)

    _qubits, edges = load_cal(args.calibration)
    kicks, series, coverage, n_leaves = read_perqubit(args.results, args.arm)
    if not series:
        raise SystemExit(
            f"no aggregated_autocorr_perqubit.dat found under {args.results} "
            f"for arm '{args.arm}'. Per-qubit data is required for this view "
            "(was the run produced with the P3 per-qubit reducer?).")

    os.makedirs(args.out, exist_ok=True)
    out_html = os.path.join(args.out, "qpu_dtc_volume_3d.html")
    path, lit, unmatched, n_sheets, vmax = write_volume_html(
        kicks, series, coverage, edges, out_html,
        arm=args.arm, stagger=args.stagger, stride=max(1, args.stride),
        opacity=args.opacity, guides=args.guides)

    print(f"arm={args.arm}  leaves={n_leaves}  kicks={len(kicks)}  "
          f"sheets drawn={n_sheets}  |intensity| max={vmax:.4f}")
    print(f"sampled qubits mapped to lattice: {len(lit)}/{len(series)}")
    if unmatched:
        print(f"WARNING: {len(unmatched)} physical_q NOT in lattice table "
              f"(skipped): {sorted(unmatched)}")
    print("wrote", path)


if __name__ == "__main__":
    main()
