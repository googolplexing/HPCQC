#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Combine two aggregate_floquet.py outputs into the F4 banked reference CSV.

Workflow (RED-RESP-§7.5 V1.1-F4; reconciles RED-VERIFY §5.2):
  1. python aggregate_floquet.py <noiseless_run_dir>      # -> aggregated_autocorr.dat
  2. python aggregate_floquet.py <device_calibrated_dir>  # -> aggregated_autocorr.dat
  3. python build_reference_csv.py \
         <noiseless_dir>/aggregated_autocorr.dat \
         <device_cal_dir>/aggregated_autocorr.dat \
         examples/reference/floquet_dtc_q10_autocorr.csv

Output columns: kick, noiseless_mean, noiseless_sem, device_cal_mean, device_cal_sem
The 'kick' column is carried through verbatim from aggregate_floquet.py's
0-indexed kick (so the BYO acceptance test must use the same convention).
"""
import os
import sys
import numpy as np


def load_agg(path):
    arr = np.loadtxt(path)  # '#' header skipped by default
    if arr.ndim != 2 or arr.shape[1] != 3:
        sys.exit(f"{path}: expected 3 columns (kick mean sem), got shape {arr.shape}")
    return arr[:, 0].astype(int), arr[:, 1], arr[:, 2]


def main():
    if len(sys.argv) != 4:
        sys.exit("usage: build_reference_csv.py "
                 "<noiseless_agg.dat> <device_cal_agg.dat> <out.csv>")
    nz_path, dc_path, out_path = sys.argv[1:4]
    k_nz, m_nz, s_nz = load_agg(nz_path)
    k_dc, m_dc, s_dc = load_agg(dc_path)
    if not np.array_equal(k_nz, k_dc):
        sys.exit(f"kick columns differ between {nz_path} and {dc_path}; "
                 f"the two runs must share the same kick grid")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("kick,noiseless_mean,noiseless_sem,device_cal_mean,device_cal_sem\n")
        for i in range(len(k_nz)):
            f.write(f"{k_nz[i]},{m_nz[i]:.6f},{s_nz[i]:.6f},"
                    f"{m_dc[i]:.6f},{s_dc[i]:.6f}\n")
    print(f"Wrote {out_path} ({len(k_nz)} kicks)")
    # Sanity sketch (matches the v1.2 §7.5.6 expectation): device-cal should be
    # well below noiseless by mid-chain.
    mid = len(k_nz) // 2
    print(f"  sanity @ kick[{k_nz[mid]}]: noiseless={m_nz[mid]:.3f}  "
          f"device_cal={m_dc[mid]:.3f}")


if __name__ == "__main__":
    main()
