# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""BYO counts observable + per-instance seed derivation (SPEC-002 §7.5 / D3.4b).

These are BYTE-IDENTICAL to the banked reference in the repo-root
``floquet_runner_v2.py`` (``get_autocorrelation`` :94-110, ``resolve_instance_seed``
:69-91). They are re-homed here (not imported) because ``floquet_runner_v2`` lives
at the repo root, which is NOT on the package path ($HPCQC_ROOT/src) — importing
it from the sweep engine would fail in-container.

This mirroring is the same discipline ``byo_sweep._spawn_rng`` already uses
(":252 — mirror floquet_runner.resolve_instance_seed"). The gate-2 reproduction
depends on these matching the bank exactly; if the bank's definitions ever
change, update here in lockstep (and re-run the reproduction).
"""

from __future__ import annotations

import numpy as np


def get_autocorrelation(counts, init_bit_array, num_qubits):
    """Autocorrelator from a counts dict. Byte-identical to
    floquet_runner_v2.get_autocorrelation: per bitstring, reverse to
    little-endian, +1 per wire matching init_bit_array else -1, weight by count,
    normalize by total_shots * num_qubits."""
    total_shots = sum(counts.values())
    num_qub = len(list(counts.keys())[0])
    total_corr = 0
    for bitstring, count in counts.items():
        plus = 0
        minus = 0
        bit_array_little = np.array(list(bitstring), dtype=int)
        bit_array = bit_array_little[::-1]
        for wire in range(num_qubits):
            if bit_array[wire] == init_bit_array[wire]:
                plus += 1
            else:
                minus += 1
        temp_corr = (plus - minus) * count
        total_corr += temp_corr
    return total_corr / (total_shots * num_qub)


def resolve_instance_seed(master_seed, instance_id):
    """Per-instance integer seed (or None for fresh entropy). Byte-identical to
    floquet_runner_v2.resolve_instance_seed: SeedSequence(master_seed).spawn so
    instances are independent yet deterministic; the single returned int seeds
    BOTH disorder draws and Aer's seed_simulator, so the instance is reproducible
    as one unit. master_seed None/"random" -> None (entropy)."""
    if master_seed is None or master_seed == "random":
        return None
    child = np.random.SeedSequence(int(master_seed)).spawn(instance_id + 1)[instance_id]
    return int(child.generate_state(1, dtype=np.uint32)[0])


def aggregate_byo_autocorr(per_seed_series, out_dir, *, write_per_instance=True):
    """Average per-instance autocorrelator vectors and write the .dat files,
    BYTE-FORMAT-IDENTICAL to the banked floquet_runner_v2 + aggregate_floquet
    chain (so the gate-2 reproduction compares like-for-like).

    Args:
      per_seed_series: list of (seed, autocorrelator_list), one per instance.
                       All vectors must have the same length (the kick grid).
      out_dir: directory to write into (created if absent).
      write_per_instance: also emit instance_NN_autocorr.dat (the bank does;
                          aggregate_floquet reads these).

    Writes:
      instance_{seed:02d}_autocorr.dat  — "# kick   autocorrelator", "{n:4d} {v:10.4f}"
      aggregated_autocorr.dat           — "# kick  mean_autocorr  sem",
                                          "{n:4d} {mean:10.4f} {sem:10.4f}"
                                          (sem = std(ddof=1)/sqrt(N), matching
                                          aggregate_floquet.py)

    Returns: (mean_corr, sem_corr) as numpy arrays.
    """
    import os

    os.makedirs(out_dir, exist_ok=True)
    mat = np.array([np.asarray(a, dtype=float) for _, a in per_seed_series])  # (N_inst, N_kicks)
    n_inst, n_kicks = mat.shape
    mean_corr = mat.mean(axis=0)
    # ddof=1 matches aggregate_floquet; for a single instance sem is undefined
    # (0/0) — emit zeros rather than nan so the .dat stays numeric.
    if n_inst > 1:
        sem_corr = mat.std(axis=0, ddof=1) / np.sqrt(n_inst)
    else:
        sem_corr = np.zeros(n_kicks)

    if write_per_instance:
        for seed, autocorr in per_seed_series:
            path = os.path.join(out_dir, f"instance_{int(seed):02d}_autocorr.dat")
            with open(path, "w", encoding="utf-8") as f:
                f.write("# kick   autocorrelator\n")
                for n, v in enumerate(autocorr):
                    f.write(f"{n:4d} {float(v):10.4f}\n")

    agg_path = os.path.join(out_dir, "aggregated_autocorr.dat")
    with open(agg_path, "w", encoding="utf-8") as f:
        f.write("# kick  mean_autocorr  sem\n")
        for n in range(n_kicks):
            f.write(f"{n:4d} {mean_corr[n]:10.4f} {sem_corr[n]:10.4f}\n")

    return mean_corr, sem_corr
