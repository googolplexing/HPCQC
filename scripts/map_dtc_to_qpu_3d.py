#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Map Floquet-DTC results onto the physical Q50 lattice (interactive 3D HTML).

For each chain (placement) in byo_dat/, compute its DTC subharmonic strength from
the autocorr arm, then attribute it to the chain's physical qubits. A qubit's
value is the MEAN over every chain that included it (coverage = #chains). The
result is a per-qubit "time-crystal surface" you can rotate, with the T2 landscape
available for correlation.

Emits:
  qpu_dtc_surface_3d.html   3D lattice: height & color = mean DTC subharmonic per
                            qubit, coverage in hover, bonds drawn, uncovered gray.
  dtc_vs_t2.html            per-qubit DTC subharmonic vs T2-echo scatter (the
                            "does coherence predict time-crystal rigidity" plot).

NOTE: until the autocorrelator is computed per-qubit (byo_observable change), each
chain contributes ONE scalar smeared across its 10 qubits, so the surface is at
chain resolution. With per-qubit data each qubit gets its own value and overlap
averaging makes it a true per-qubit surface (this script reads per-qubit .dat
automatically if present).

  python map_dtc_to_qpu_3d.py examples/q50_calibration_20260524_08c3c70f.json \\
      results/autocorr_echo_12p_10qb_10s_40k_1000s_19055840 --metric t2_echo_us
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np

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
    c, b = LAYOUT[qb]
    return c, -b


def load_cal(path):
    d = json.load(open(path))
    return d["qubits"], d.get("qubit_connectivity", [])


def subharmonic_amp(values):
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return 0.0
    X = np.fft.rfft(v - v.mean())
    return float(2.0 * np.abs(X[-1]) / n)


def per_qubit_dtc(results_dir, per_qubit=False):
    """Map DTC subharmonic onto qubits. Returns {QB: (mean_dtc, coverage)}.

    Default (``per_qubit=False``, legacy): each chain contributes ONE scalar
    smeared across its physical qubits — a chain-resolution surface.

    ``per_qubit=True``: read the self-describing per-qubit .dat
    (``aggregated_autocorr_perqubit.dat``), compute each qubit's OWN subharmonic
    from its own column, and attribute it by the file's ``physical_q`` column —
    a true site-resolved surface (RED-RULING-PER-QUBIT §2.1). FAILS LOUD if
    per-qubit is requested but a chain has only the chain-averaged scalar:
    silently smearing chain data as per-site is the exact failure this mode
    exists to remove (RED-RULING-PER-QUBIT D6 / condition 4).
    """
    sums = {}
    counts = {}
    for root, _dirs, files in os.walk(results_dir):
        parts = os.path.normpath(root).split(os.sep)
        if len(parts) < 3:
            continue
        obs, _env, phys = parts[-1], parts[-2], parts[-3]
        if obs != "autocorr":
            continue
        has_scalar = "aggregated_autocorr.dat" in files
        has_pq = "aggregated_autocorr_perqubit.dat" in files
        if per_qubit:
            if not has_pq:
                if not has_scalar:
                    continue
                raise SystemExit(
                    f"per-qubit surface requested but {root} has only the "
                    f"chain-averaged aggregated_autocorr.dat (no "
                    f"aggregated_autocorr_perqubit.dat). Refusing to smear chain "
                    f"data as per-site (RED-RULING-PER-QUBIT D6). Re-run with the "
                    f"per-qubit observable, or drop --per-qubit for the smeared "
                    f"chain-resolution surface."
                )
            # Self-describing: attribute each qubit by the file's physical_q
            # column, NOT by re-parsing the path, so a re-sorted placement cannot
            # transpose sites (RED §2.1). Columns: kick local_q physical_q mean sem.
            series = {}  # physical_q -> list of (kick, mean)
            with open(os.path.join(root, "aggregated_autocorr_perqubit.dat")) as fh:
                for ln in fh:
                    if ln.startswith("#") or not ln.strip():
                        continue
                    c = ln.split()
                    kick, physical_q, mean = int(c[0]), c[2], float(c[3])
                    series.setdefault(physical_q, []).append((kick, mean))
            for q, kv in series.items():
                vals = [m for _, m in sorted(kv)]
                s = subharmonic_amp(vals)
                sums[q] = sums.get(q, 0.0) + s
                counts[q] = counts.get(q, 0) + 1
        else:
            if not has_scalar:
                continue
            qbs = re.findall(r"QB\d+", phys)
            if not qbs:
                continue
            arr = np.loadtxt(
                os.path.join(root, "aggregated_autocorr.dat"), comments="#"
            )
            if arr.ndim == 1:
                arr = arr[None, :]
            chain_dtc = subharmonic_amp(arr[:, 1])
            for q in qbs:  # smear chain scalar over its physical qubits
                sums[q] = sums.get(q, 0.0) + chain_dtc
                counts[q] = counts.get(q, 0) + 1
    return {q: (sums[q] / counts[q], counts[q]) for q in sums}


def write_surface_html(qubits, edges, dtc, path):
    import plotly.graph_objects as go
    vals = [v for v, _ in dtc.values()]
    vmin, vmax = (min(vals), max(vals)) if vals else (0.0, 1.0)
    span = (vmax - vmin) or 1.0
    ex, ey, ez = [], [], []
    znorm = lambda q: (dtc[q][0] - vmin) / span if q in dtc else 0.0
    for a, b in edges:
        if a in LAYOUT and b in LAYOUT:
            xa, ya = xy(a); xb, yb = xy(b)
            ex += [xa, xb, None]; ey += [ya, yb, None]; ez += [znorm(a), znorm(b), None]
    cx, cy, cz, cc, ct = [], [], [], [], []
    ux, uy, ut = [], [], []
    for q in LAYOUT:
        x, y = xy(q)
        if q in dtc:
            v, cov = dtc[q]
            cx.append(x); cy.append(y); cz.append((v - vmin) / span); cc.append(v)
            t2 = qubits.get(q, {}).get("t2_echo_us", float("nan"))
            ct.append(f"{q}<br>DTC subharm={v:.3f}<br>coverage={cov} chains<br>T2echo={t2:.1f} us")
        else:
            ux.append(x); uy.append(y); ut.append(f"{q}<br>not sampled")
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=ex, y=ey, z=ez, mode="lines",
                               line=dict(color="gray", width=3), hoverinfo="skip"))
    fig.add_trace(go.Scatter3d(
        x=cx, y=cy, z=cz, mode="markers+text",
        marker=dict(size=8, color=cc, colorscale="Plasma",
                    colorbar=dict(title="DTC subharm"), cmin=vmin, cmax=vmax,
                    line=dict(width=0.5, color="black")),
        text=[q.replace("QB", "") for q in LAYOUT if q in dtc],
        textposition="top center", textfont=dict(size=8),
        hovertext=ct, hoverinfo="text", name="DTC"))
    if ux:
        fig.add_trace(go.Scatter3d(x=ux, y=uy, z=[0] * len(ux), mode="markers",
                                   marker=dict(size=6, color="lightgray", symbol="x"),
                                   hovertext=ut, hoverinfo="text", name="not sampled"))
    fig.update_layout(
        title="Q50 time-crystal surface — height & color = DTC subharmonic strength",
        scene=dict(xaxis_title="lattice column", yaxis_title="lattice band",
                   zaxis_title="DTC subharmonic (normalized)",
                   aspectmode="manual", aspectratio=dict(x=1.4, y=1.4, z=0.7)))
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def write_corr_html(qubits, dtc, path):
    import plotly.graph_objects as go
    xs, ys, ts = [], [], []
    for q, (v, cov) in dtc.items():
        t2 = qubits.get(q, {}).get("t2_echo_us")
        if t2 is not None:
            xs.append(t2); ys.append(v); ts.append(f"{q} (cov {cov})")
    fig = go.Figure(go.Scatter(x=xs, y=ys, mode="markers+text", text=[t.split()[0] for t in ts],
                               textposition="top center", hovertext=ts, hoverinfo="text",
                               marker=dict(size=9)))
    fig.update_layout(title="Per-qubit DTC subharmonic vs T2-echo",
                      xaxis_title="T2-echo (us)", yaxis_title="DTC subharmonic")
    fig.write_html(path, include_plotlyjs=True, full_html=True)
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calibration")
    ap.add_argument("results")
    ap.add_argument("--metric", default="t2_echo_us")
    ap.add_argument("--out", default="qpu_dtc_map")
    ap.add_argument(
        "--per-qubit", action="store_true",
        help="true site-resolved surface from aggregated_autocorr_perqubit.dat "
             "(fails loud if absent); default smears the chain scalar",
    )
    args = ap.parse_args(argv)

    qubits, edges = load_cal(args.calibration)
    dtc = per_qubit_dtc(args.results, per_qubit=args.per_qubit)
    os.makedirs(args.out, exist_ok=True)
    w = [write_surface_html(qubits, edges, dtc,
                            os.path.join(args.out, "qpu_dtc_surface_3d.html")),
         write_corr_html(qubits, dtc, os.path.join(args.out, "dtc_vs_t2.html"))]
    cov = [c for _, c in dtc.values()]
    print(f"sampled qubits: {len(dtc)}/{len(LAYOUT)}  "
          f"coverage min/mean/max = {min(cov)}/{sum(cov)/len(cov):.1f}/{max(cov)}")
    for p in w:
        print("wrote", p)


if __name__ == "__main__":
    main()
