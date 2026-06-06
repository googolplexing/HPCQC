# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Consumer-side ordering test for the per-qubit surface reader
(RED-RULING-PER-QUBIT §2.1, condition 1).

The writer-level self-describing/ordering guarantee is unit-tested in
test_persite_output.py / test_byo_autocorr_perqubit.py; this is the END-TO-END
version Red required: emit a per-qubit .dat for a deliberately non-monotonic
placement where exactly one site carries the period-2 signal, read it back
THROUGH map_dtc_to_qpu_3d.per_qubit_dtc, and assert the subharmonic lands on the
right PHYSICAL qubit (a transposing reader would put it on the wrong site).
Also asserts the fail-loud behavior when per-qubit is requested but absent.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPT = os.path.join(_HERE, "..", "scripts", "map_dtc_to_qpu_3d.py")
_spec = importlib.util.spec_from_file_location("map_dtc_to_qpu_3d", _SCRIPT)
mdq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mdq)


def _leaf(results_dir, phys):
    d = os.path.join(results_dir, "floquet", phys, "device_calibrated", "autocorr")
    os.makedirs(d, exist_ok=True)
    return d


def _write_perqubit(d, phys_ids, signal_local, nkicks=8):
    """Per-qubit .dat where only local index ``signal_local`` carries (-1)^k
    (a perfect period-2 -> high subharmonic); all other sites are flat (S=0)."""
    with open(os.path.join(d, "aggregated_autocorr_perqubit.dat"), "w") as f:
        f.write("# kick  local_q  physical_q  mean_autocorr  sem\n")
        for k in range(nkicks):
            for lq, pid in enumerate(phys_ids):
                v = ((-1.0) ** k) if lq == signal_local else 0.0
                f.write(f"{k:4d} {lq:4d} {pid:>10} {v:10.4f} {0.0:10.4f}\n")


def test_reader_attributes_signal_to_correct_physical_qubit(tmp_path):
    results = str(tmp_path)
    phys = ["QB5", "QB42", "QB13"]  # non-monotonic; signal on local index 1 = QB42
    _write_perqubit(_leaf(results, "-".join(phys)), phys, signal_local=1)

    dtc = mdq.per_qubit_dtc(results, per_qubit=True)

    # The period-2 site (QB42) must carry the high subharmonic; the flat sites ~0.
    s_qb42 = dtc["QB42"][0]
    assert s_qb42 > 1.5, f"expected strong subharmonic on QB42, got {s_qb42}"
    assert dtc["QB5"][0] < 1e-9 and dtc["QB13"][0] < 1e-9
    # not transposed onto a neighbor
    assert dtc["QB42"][0] > dtc["QB5"][0] and dtc["QB42"][0] > dtc["QB13"][0]


def test_reader_fails_loud_when_perqubit_requested_but_absent(tmp_path):
    results = str(tmp_path)
    d = _leaf(results, "QB5-QB42-QB13")
    # only the chain-averaged scalar present (no per-qubit file)
    with open(os.path.join(d, "aggregated_autocorr.dat"), "w") as f:
        f.write("# kick  mean_autocorr  sem\n")
        for k in range(8):
            f.write(f"{k:4d} {((-1.0)**k):10.4f} {0.0:10.4f}\n")
    with pytest.raises(SystemExit):
        mdq.per_qubit_dtc(results, per_qubit=True)


def test_legacy_smear_path_unchanged(tmp_path):
    """Default (per_qubit=False) still smears the chain scalar over its qubits."""
    results = str(tmp_path)
    d = _leaf(results, "QB5-QB42-QB13")
    with open(os.path.join(d, "aggregated_autocorr.dat"), "w") as f:
        f.write("# kick  mean_autocorr  sem\n")
        for k in range(8):
            f.write(f"{k:4d} {((-1.0)**k):10.4f} {0.0:10.4f}\n")
    dtc = mdq.per_qubit_dtc(results, per_qubit=False)
    # all three qubits get the SAME chain scalar (smeared)
    vals = [dtc[q][0] for q in ("QB5", "QB42", "QB13")]
    assert max(vals) - min(vals) < 1e-9 and vals[0] > 1.5
