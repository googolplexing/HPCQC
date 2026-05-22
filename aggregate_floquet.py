#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""
Aggregate per-instance Floquet results.

Reads every ``instance_*_autocorr.dat`` in the given directory,
averages the autocorrelator vectors across gate instances (matches the
original script's ``autocorrelators / num_gate_instances`` step), then
reproduces the two plots from the original script (autocorrelator vs
driving period; FFT amplitude spectrum). Also emits the standard error
of the mean across instances as a per-kick uncertainty band.

Usage:
  python aggregate_floquet.py <results_dir>
"""
import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless on LUMI compute nodes
import matplotlib.pyplot as plt


if len(sys.argv) != 2:
    sys.exit("usage: python aggregate_floquet.py <results_dir>")

results_dir = sys.argv[1]
dat_files = sorted(glob.glob(os.path.join(results_dir, "instance_*_autocorr.dat")))
if not dat_files:
    sys.exit(f"no instance_*_autocorr.dat files found in {results_dir}")

print(f"Found {len(dat_files)} instance files in {results_dir}")

per_instance = []
for path in dat_files:
    arr = np.loadtxt(path)
    per_instance.append(arr[:, 1])
    print(f"  {os.path.basename(path)}: {arr.shape[0]} kicks")

per_instance   = np.array(per_instance)             # shape (N_inst, N_kicks)
mean_corr      = per_instance.mean(axis=0)
sem_corr       = per_instance.std(axis=0, ddof=1) / np.sqrt(per_instance.shape[0])
num_max_kicks  = mean_corr.shape[0]

# Aggregated .dat (same format as original, with extra SEM column)
agg_path = os.path.join(results_dir, "aggregated_autocorr.dat")
with open(agg_path, "w", encoding="utf-8") as f:
    f.write("# kick  mean_autocorr  sem\n")
    for n in range(num_max_kicks):
        f.write(f"{n:4d} {mean_corr[n]:10.4f} {sem_corr[n]:10.4f}\n")
print(f"\nWrote {agg_path}")

# ───────── FFT (identical to the original script) ─────────
fft_result = np.fft.fft(mean_corr)
fs = len(fft_result) + 1
T  = 1.0 / fs
N  = len(fft_result)
t  = np.linspace(0.0, N * T, N)
y  = np.abs(fft_result) / np.sum(np.abs(fft_result))

# ───────── autocorrelator plot ─────────
plt.rcParams.update({"font.size": 14})

fig, ax = plt.subplots(figsize=(8, 5))
ax.set_ylim(-1, 1)
ax.errorbar(
    range(1, num_max_kicks + 1), mean_corr, yerr=sem_corr,
    fmt="o-", capsize=3,
    label=f"mean of {len(dat_files)} gate instances",
)
ax.set_xlabel("Driving Periods ($T$ steps)")
ax.set_ylabel(r"$\langle A(0)\,A(T)\rangle$")
ax.grid(True)
ax.legend()
fig.tight_layout()
fig_path = os.path.join(results_dir, "autocorrelator.png")
fig.savefig(fig_path, dpi=150)
plt.close(fig)
print(f"Wrote {fig_path}")

# ───────── FFT spectrum plot ─────────
fig, ax = plt.subplots(figsize=(8, 5))
ax.set_xlim(0.35, 0.65)
ax.plot(t, y, "o-")
ax.set_xlabel("Frequency (1/T)")
ax.set_ylabel("Amplitude")
ax.grid(True)
fig.tight_layout()
fft_path = os.path.join(results_dir, "fft_amplitude.png")
fig.savefig(fft_path, dpi=150)
plt.close(fig)
print(f"Wrote {fft_path}")

print("\nDone.")
