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
