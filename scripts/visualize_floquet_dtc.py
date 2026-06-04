#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Visualize a Floquet-DTC BYO echo sweep from its byo_dat/ .dat tree.

Reads the merged results directory (the one with byo_dat/ + sweep.h5) and emits,
into <results>/figures/ :

  placement_<phys>.png          one per placement: autocorr + echo, mean +/- SEM
                                over seeds, with the 10 seed traces faint behind.
  grid_autocorr_staggered.png   12-panel small-multiples of the *staggered*
                                envelope (-1)^kick * A(kick) -- the DTC rigidity
                                view (turns the +,-,+,- zigzag into a decay curve).
  subharmonic_by_placement.png  the headline quantitative figure: period-2 (f=1/2)
                                Fourier amplitude per placement, with per-seed
                                scatter. One number per placement = "how good a DTC".
  heatmap_autocorr.png          placement x kick heatmap of mean A -- compact
                                all-at-a-glance overview.

Pure numpy + matplotlib; no h5py. Run inside the qiskit container:

  python visualize_floquet_dtc.py /flash/.../results/autocorr_echo_12p_..._19055840
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless on a compute/login node
import matplotlib.pyplot as plt


def _load_dat(path):
    """Load a 2-col (kick, value) or 3-col (kick, mean, sem) .dat -> (kick, cols...)."""
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:  # single row
        arr = arr[None, :]
    return arr


def discover(results_dir):
    """Walk the tree; group aggregated_autocorr.dat by placement, keyed by observable.

    Path tail is .../{phys}/{env}/{obs}/aggregated_autocorr.dat (two observables ->
    obs is a real dir). We parse positionally from the leaf upward so we don't
    depend on where byo_dat/{stem} sits.
    Returns: {phys: {obs: {"agg": Nx3 array, "seeds": [Nx2 arrays]}}}
    """
    out = defaultdict(dict)
    for root, _dirs, files in os.walk(results_dir):
        if "aggregated_autocorr.dat" not in files:
            continue
        parts = os.path.normpath(root).split(os.sep)
        if len(parts) < 3:
            continue
        obs, env, phys = parts[-1], parts[-2], parts[-3]
        # Single-observable runs have no obs dir; then leaf is the env dir.
        if obs == env:  # defensive
            obs = "autocorr"
        agg = _load_dat(os.path.join(root, "aggregated_autocorr.dat"))
        seeds = []
        for fn in sorted(files):
            if fn.startswith("instance_") and fn.endswith("_autocorr.dat"):
                seeds.append(_load_dat(os.path.join(root, fn)))
        out[phys][obs] = {"agg": agg, "seeds": seeds, "env": env}
    return out


def subharmonic_amp(values):
    """Period-2 (f=1/2, Nyquist) Fourier amplitude of a kick series.

    A DTC has A(kick) ~ (-1)^kick * envelope, so the discrete spectrum concentrates
    at the Nyquist bin. Return its peak-to-peak amplitude 2*|X_Nyq|/N -- a single
    scalar 'subharmonic strength' per series.
    """
    v = np.asarray(values, dtype=float)
    n = len(v)
    if n < 2:
        return 0.0
    X = np.fft.rfft(v - v.mean())  # drop DC; we want the f=1/2 weight
    nyq = X[-1] if n % 2 == 0 else X[-1]  # last rfft bin ~ f=1/2 for even n
    return float(2.0 * np.abs(nyq) / n)


def plot_placement(phys, obsmap, outdir):
    observables = [o for o in ("autocorr", "echo") if o in obsmap] or list(obsmap)
    fig, axes = plt.subplots(1, len(observables), figsize=(6 * len(observables), 4.2),
                             squeeze=False)
    for ax, obs in zip(axes[0], observables):
        d = obsmap[obs]
        kick = d["agg"][:, 0]
        mean = d["agg"][:, 1]
        sem = d["agg"][:, 2] if d["agg"].shape[1] > 2 else np.zeros_like(mean)
        for s in d["seeds"]:
            ax.plot(s[:, 0], s[:, 1], color="0.75", lw=0.6, zorder=1)
        ax.plot(kick, mean, color="C0", lw=1.8, zorder=3, label="mean over seeds")
        ax.fill_between(kick, mean - sem, mean + sem, color="C0", alpha=0.25,
                        zorder=2, label="+/- SEM")
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_title(f"{obs}")
        ax.set_xlabel("Floquet kick")
        ax.set_ylabel("autocorrelator")
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(f"placement {phys}", fontsize=11)
    fig.tight_layout()
    p = os.path.join(outdir, f"placement_{phys}.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_grid_staggered(data, outdir):
    phys_list = sorted(data)
    n = len(phys_list)
    ncol = 4
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.4 * nrow),
                             sharex=True, sharey=True, squeeze=False)
    for i, phys in enumerate(phys_list):
        ax = axes[i // ncol][i % ncol]
        if "autocorr" not in data[phys]:
            continue
        agg = data[phys]["autocorr"]["agg"]
        kick = agg[:, 0]
        stag = ((-1.0) ** kick) * agg[:, 1]  # staggered: zigzag -> envelope
        ax.plot(kick, stag, color="C3", lw=1.5)
        ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        ax.set_title(phys, fontsize=7)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(r"Staggered autocorrelator  $(-1)^{kick}\,A(kick)$  (DTC rigidity)",
                 fontsize=12)
    fig.supxlabel("Floquet kick")
    fig.tight_layout()
    p = os.path.join(outdir, "grid_autocorr_staggered.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_subharmonic(data, outdir):
    phys_list = sorted(data)
    mean_amp, seed_amps = [], []
    for phys in phys_list:
        if "autocorr" not in data[phys]:
            mean_amp.append(0.0); seed_amps.append([]); continue
        d = data[phys]["autocorr"]
        mean_amp.append(subharmonic_amp(d["agg"][:, 1]))
        seed_amps.append([subharmonic_amp(s[:, 1]) for s in d["seeds"]])
    x = np.arange(len(phys_list))
    fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(phys_list)), 4.5))
    ax.bar(x, mean_amp, color="C0", alpha=0.7, label="mean-series subharmonic amp")
    for xi, amps in zip(x, seed_amps):
        if amps:
            ax.scatter([xi] * len(amps), amps, s=14, color="C3", zorder=3,
                       alpha=0.8)
    ax.scatter([], [], s=14, color="C3", label="per-seed")
    ax.set_xticks(x)
    ax.set_xticklabels(phys_list, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel(r"period-2 amplitude $2|X_{1/2}|/N$")
    ax.set_title("DTC subharmonic strength by placement (higher = more rigid)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    p = os.path.join(outdir, "subharmonic_by_placement.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def plot_heatmap(data, outdir):
    phys_list = sorted(p for p in data if "autocorr" in data[p])
    if not phys_list:
        return None
    mat = np.array([data[p]["autocorr"]["agg"][:, 1] for p in phys_list])
    kick = data[phys_list[0]]["autocorr"]["agg"][:, 0]
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(phys_list) + 1.5))
    vmax = np.nanmax(np.abs(mat)) or 1.0
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   extent=[kick.min(), kick.max(), len(phys_list) - 0.5, -0.5])
    ax.set_yticks(range(len(phys_list)))
    ax.set_yticklabels(phys_list, fontsize=7)
    ax.set_xlabel("Floquet kick")
    ax.set_title("Mean autocorrelator: placement x kick")
    fig.colorbar(im, ax=ax, label="autocorrelator")
    fig.tight_layout()
    p = os.path.join(outdir, "heatmap_autocorr.png")
    fig.savefig(p, dpi=140)
    plt.close(fig)
    return p


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="merged results dir (contains byo_dat/)")
    ap.add_argument("--out", default=None, help="figure dir (default: <results>/figures)")
    args = ap.parse_args(argv)

    data = discover(args.results_dir)
    if not data:
        sys.exit(f"no aggregated_autocorr.dat found under {args.results_dir}")
    outdir = args.out or os.path.join(args.results_dir, "figures")
    os.makedirs(outdir, exist_ok=True)

    written = []
    for phys in sorted(data):
        written.append(plot_placement(phys, data[phys], outdir))
    written.append(plot_grid_staggered(data, outdir))
    written.append(plot_subharmonic(data, outdir))
    hm = plot_heatmap(data, outdir)
    if hm:
        written.append(hm)

    print(f"{len(data)} placement(s), {sum(len(v) for v in data.values())} "
          f"(placement,observable) series")
    for p in written:
        print("  wrote", p)


if __name__ == "__main__":
    main()
