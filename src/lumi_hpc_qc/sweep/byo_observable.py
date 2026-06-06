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

# Reserved name for the synthesized single-observable case (absent ``observables``
# in the YAML). It is the only observable name that maps to the LEGACY path
# layout, so existing single-observable runs (incl. the W1.6 gate) stay
# byte-identical. Any declared observable carries its own name and gets an extra
# path level. See byo_observable_subpath.
DEFAULT_OBSERVABLE_NAME = "default"


def byo_observable_subpath(
    observable_name: str,
    circuit_function: str | None = None,
    disambiguate_default: bool = False,
) -> str:
    """Trailing path segment for a BYO observable's HDF5 group / .dat subdir.

    Single source of the legacy-vs-observable layout decision, consumed by BOTH
    the .dat aggregator (sweep_engine `_execute_byo_group`) and the HDF5 writer
    (`SweepHDF5Writer.write_byo_result`) so the two cannot drift:

      - a declared observable -> "/<name>" (one extra level), so multiple
        families under the same (placement, env) do not collide;
      - the synthesized default observable ("default") -> "" (LEGACY layout;
        the path is exactly what the single-observable path produced pre-D7, so
        banked references and the W1.6 gate are untouched) -- EXCEPT when the
        resolved run has >1 circuit family sharing this leaf.

    BYO-FAMILY-COLLISION fix (b1): when the caller has determined (at run level,
    across groups) that more than one circuit family resolves to the same
    (script_stem, placement, env) leaf, it sets ``disambiguate_default=True``
    and ALL colliding families -- including the default one -- take a
    "/<circuit_function>" segment. This is all-families-or-none, keyed on the
    resolved run: the default family's path must NOT depend on whether some
    other family happens to exist (that would be order-dependent and fragile).
    The one-family case is the only "" case, so single-arm runs stay byte-
    identical to the bank.
    """
    if observable_name != DEFAULT_OBSERVABLE_NAME:
        return f"/{observable_name}"
    if disambiguate_default and circuit_function:
        return f"/{circuit_function}"
    return ""


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


def get_autocorrelation_perqubit(counts, init_bit_array, num_qubits):
    """Per-qubit (un-collapsed) autocorrelator vector from a counts dict.

    The site-resolved form of ``get_autocorrelation`` (RED-RULING-PER-QUBIT §1).
    Same little-endian reversal and match-vs-init convention; per wire i it
    returns ``A_i = <s_i>`` where ``s_i = +1`` when the measured bit at wire i
    equals ``init_bit_array[i]`` else ``-1``, weighted by count and normalized by
    ``total_shots``. Wire i is the logical qubit; its physical qubit is
    ``physical_qubit_set[i]`` (path order — load-bearing, see the .dat writer).

    Parity (the value-shape-only invariant): the legacy scalar is the mean over
    wires — ``get_autocorrelation(...) == perqubit.sum() / num_qub`` exactly, and
    equals ``perqubit.mean()`` when the measured bitstring width ``num_qub``
    equals ``num_qubits`` (the chain is measured exactly, which is the validated
    path). The integer per-wire sums here are the same terms ``get_autocorrelation``
    accumulates into ``total_corr`` before its single division, so nothing is
    discarded — the un-collapse is free in compute.

    Returns: numpy array of shape (num_qubits,).
    """
    total_shots = sum(counts.values())
    init = np.asarray(init_bit_array)[:num_qubits]
    corr = np.zeros(num_qubits, dtype=float)
    for bitstring, count in counts.items():
        bit_array = np.array(list(bitstring), dtype=int)[::-1][:num_qubits]
        s = np.where(bit_array == init, 1.0, -1.0)
        corr += s * count
    return corr / total_shots


def aggregate_byo_autocorr_perqubit(per_seed_matrices, physical_qubit_set, out_dir):
    """Write the SELF-DESCRIBING per-qubit autocorrelator .dat (RED-RULING-PER-QUBIT
    D1, §2.1) — a thin, autocorr-named wrapper over the generic per-site
    serializer ``data.persite_output.write_persite_series``.

    The generic helper is the reusable seam: it does the instance-axis mean/sem
    (preserving the site axis automatically, RED §1) and the self-describing
    write (physical qubit id carried in the file). This wrapper just pins the
    autocorr filename and the RED-D1 column names (``kick local_q physical_q``),
    so new per-site observables (per-site polarization, classical-shadow per-site
    estimates, correlation-length's sibling) reuse ``write_persite_series``
    directly instead of re-implementing the format. Local import keeps this
    module qiskit-free and avoids any cross-package import-time coupling.

    Args:
      per_seed_matrices: list of ``(seed, matrix)``; each matrix has shape
                         ``(N_kicks, num_qubits)`` (per-qubit vectors per kick,
                         e.g. stacked ``get_autocorrelation_perqubit``).
      physical_qubit_set: ordered logical->physical qubit ids, length num_qubits.
      out_dir: directory to write into (created if absent).

    Writes ``aggregated_autocorr_perqubit.dat``; returns (mean, sem) arrays of
    shape (N_kicks, num_qubits).
    """
    from lumi_hpc_qc.data.persite_output import write_persite_series

    return write_persite_series(
        per_seed_matrices,
        physical_qubit_set,
        out_dir,
        filename="aggregated_autocorr_perqubit.dat",
        grid_label="kick",
        local_label="local_q",
        physical_label="physical_q",
        value_label="autocorr",
    )
