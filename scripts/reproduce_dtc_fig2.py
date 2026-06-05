#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Reproduce Fig. 2(a-d) of Mi et al. "Time-Crystalline Eigenstate Order on a
Quantum Processor" (arXiv:2105.06632) from an HPCQC Floquet-DTC echo sweep.

Reads the merged results directory (the byo_dat/ tree) and emits a 2x2 panel that
mirrors that paper's (a-d), built from the qubit- and seed-averaged autocorrelator
of every chain in the sweep:

  (a) device-averaged autocorrelator A(t) vs Floquet cycle t (raw, oscillating).
  (b) decay-corrected A(t) / sqrt(|A_echo(t)|): the echo arm divides out the
      overall decoherence envelope (the project's defined normalization), leaving
      the intrinsic period-2 response with a flat envelope for an ideal DTC.
  (c) magnitude spectrum |FFT| of (a): the f = 1/2 subharmonic peak.
  (d) magnitude spectrum of (b).

Each chain's autocorr arm gives A_chain(t) and its echo arm A_echo,chain(t), both
already averaged over the chain's 10 physical qubits and over the 10 disorder
seeds. The device curve is the mean over chains; the decay correction is applied
*per chain* before averaging.

Differences from the paper, stated for honesty: (i) this is a calibrated-noise
AerSimulator run, not hardware, and is NOT readout-error-mitigated; (ii) our
ensemble is 200 ten-qubit chains spread across the chip rather than one fixed
chain; (iii) epsilon is fixed at 0.03 (DTC side), so there is no thermal curve to
overlay -- that needs an epsilon=0.5 run. Panels (e,f) (site-resolved) need the
per-qubit observable and are not produced here.

Pure numpy + matplotlib; no h5py. Run inside the qiskit container:
  python reproduce_dtc_fig2.py \
      /flash/.../results/autocorr_echo_12p_10qb_10s_40k_1000s_spatial_survey19064225
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless on a login/compute node
import matplotlib.pyplot as plt

ECHO_FLOOR = 1e-2  # clip |echo| below this before sqrt to avoid late-time blowup


def _load_dat(path):
    arr = np.loadtxt(path, comments="#")
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def discover(results_dir):
    """{phys: {'autocorr': Nx3, 'echo': Nx3}} from aggregated_autocorr.dat leaves.

    Path tail is .../{phys}/{env}/{obs}/aggregated_autocorr.dat; obs is the arm.
    """
    out = defaultdict(dict)
    for root, _dirs, files in os.walk(results_dir):
        if "aggregated_autocorr.dat" not in files:
            continue
        parts = os.path.normpath(root).split(os.sep)
        if len(parts) < 3:
            continue
        obs, _env, phys = parts[-1], parts[-2], parts[-3]
        if obs not in ("autocorr", "echo"):
            continue
        out[phys][obs] = _load_dat(os.path.join(root, "aggregated_autocorr.dat"))
    return out


def assemble(data):
    """Return kick axis, per-chain A matrix, per-chain corrected matrix (or None)."""
    phys = sorted(p for p in data if "autocorr" in data[p])
    if not phys:
        sys.exit("no autocorr arm found under results dir")
    kick = data[phys[0]]["autocorr"][:, 0]
    A = np.array([data[p]["autocorr"][:, 1] for p in phys])
    corr_rows, n_echo = [], 0
    for p in phys:
        if "echo" in data[p]:
            e = data[p]["echo"][:, 1]
            denom = np.sqrt(np.clip(np.abs(e), ECHO_FLOOR, None))
            corr_rows.append(data[p]["autocorr"][:, 1] / denom)
            n_echo += 1
    corr = np.array(corr_rows) if corr_rows else None
    return kick, A, corr, len(phys), n_echo


def spectrum(sig):
    """Single-sided magnitude spectrum 2|X(f)|/N of a kick series (DC removed)."""
    n = len(sig)
    X = np.fft.rfft(sig - np.mean(sig))
    f = np.fft.rfftfreq(n, d=1.0)
    amp = 2.0 * np.abs(X) / n
    return f, amp


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dir", help="merged results dir (contains byo_dat/)")
    ap.add_argument("--out", default=None, help="figure dir (default <results>/figures_fig2)")
    ap.add_argument("--title", default=None, help="optional suptitle")
    args = ap.parse_args(argv)

    data = discover(args.results_dir)
    kick, A, corr, n_ch, n_echo = assemble(data)
    outdir = args.out or os.path.join(args.results_dir, "figures_fig2")
    os.makedirs(outdir, exist_ok=True)

    A_mean = A.mean(axis=0)
    fa, sa = spectrum(A_mean)
    if corr is not None:
        C_mean = corr.mean(axis=0)
        fc, sc = spectrum(C_mean)

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))
    (a_ax, b_ax), (c_ax, d_ax) = ax

    # (a) raw averaged autocorrelator
    for row in A:
        a_ax.plot(kick, row, color="0.8", lw=0.5, zorder=1)
    a_ax.plot(kick, A_mean, color="C0", lw=2.0, zorder=3)
    a_ax.axhline(0, color="k", lw=0.5, alpha=0.4)
    a_ax.set_title(f"(a) averaged autocorrelator  ($N_{{chains}}$={n_ch})")
    a_ax.set_xlabel("Floquet cycle t"); a_ax.set_ylabel(r"$\langle A(t)\rangle$")

    # (b) decay-corrected
    if corr is not None:
        for row in corr:
            b_ax.plot(kick, row, color="0.8", lw=0.5, zorder=1)
        b_ax.plot(kick, C_mean, color="C3", lw=2.0, zorder=3)
        b_ax.axhline(0, color="k", lw=0.5, alpha=0.4)
        b_ax.set_title(f"(b) decay-corrected  $A/\\sqrt{{|A_{{echo}}|}}$  "
                       f"({n_echo}/{n_ch} chains)")
    else:
        b_ax.text(0.5, 0.5, "echo arm not found\n(no decay correction)",
                  ha="center", va="center", transform=b_ax.transAxes)
        b_ax.set_title("(b) decay-corrected  -- unavailable")
    b_ax.set_xlabel("Floquet cycle t"); b_ax.set_ylabel(r"$\langle A(t)\rangle / \sqrt{|A_{echo}|}$")

    # (c) spectrum of (a)
    c_ax.plot(fa, sa, color="C0", lw=1.5, marker="o", ms=3)
    c_ax.axvline(0.5, color="0.6", ls="--", lw=1)
    nyq_a = sa[-1]
    c_ax.annotate(f"f=1/2\n{nyq_a:.3f}", xy=(0.5, nyq_a),
                  xytext=(0.34, nyq_a * 0.85), fontsize=9)
    c_ax.set_title("(c) spectrum of (a)")
    c_ax.set_xlabel("frequency f (1/cycle)"); c_ax.set_ylabel(r"$2|X(f)|/N$")

    # (d) spectrum of (b)
    if corr is not None:
        d_ax.plot(fc, sc, color="C3", lw=1.5, marker="o", ms=3)
        d_ax.axvline(0.5, color="0.6", ls="--", lw=1)
        nyq_c = sc[-1]
        d_ax.annotate(f"f=1/2\n{nyq_c:.3f}", xy=(0.5, nyq_c),
                      xytext=(0.34, nyq_c * 0.85), fontsize=9)
        d_ax.set_title("(d) spectrum of (b)")
    else:
        d_ax.text(0.5, 0.5, "unavailable", ha="center", va="center",
                  transform=d_ax.transAxes)
        d_ax.set_title("(d) spectrum of (b) -- unavailable")
    d_ax.set_xlabel("frequency f (1/cycle)"); d_ax.set_ylabel(r"$2|X(f)|/N$")

    if args.title:
        fig.suptitle(args.title, fontsize=13)
    fig.tight_layout()
    figpath = os.path.join(outdir, "fig2_autocorr_spectra.png")
    fig.savefig(figpath, dpi=160)
    plt.close(fig)

    # also dump the underlying device-mean curves for the physicist
    datpath = os.path.join(outdir, "fig2_device_mean.dat")
    cols = [kick, A_mean]
    header = "kick  A_mean"
    if corr is not None:
        cols.append(C_mean); header += "  A_corrected"
    np.savetxt(datpath, np.c_[tuple(cols)], header=header,
               fmt=["%4d"] + ["%12.6f"] * (len(cols) - 1))

    print(f"chains: {n_ch} (autocorr), {n_echo} with echo")
    print(f"(a) f=1/2 subharmonic amplitude (raw):       {nyq_a:.4f}")
    if corr is not None:
        print(f"(b) f=1/2 subharmonic amplitude (corrected): {sc[-1]:.4f}")
    print("wrote", figpath)
    print("wrote", datpath)


if __name__ == "__main__":
    main()
