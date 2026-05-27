# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""D3.4c — BYO results storage (Option A): native HDF5 group + .dat aggregation.

The BYO counts→autocorrelator is a per-kick VECTOR, stored as its own /byo
group tree (not forced through energy-shaped SweepResultEntry), plus a
.dat/aggregated_autocorr.dat matching aggregate_floquet.py — the form the
gate-2 reproduction compares.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from lumi_hpc_qc.sweep.byo_observable import aggregate_byo_autocorr

h5py = pytest.importorskip("h5py")
from lumi_hpc_qc.data.hdf5_writer import SweepHDF5Writer


# --------------------------- aggregator format ---------------------------

def test_aggregated_dat_matches_aggregate_floquet_format():
    """aggregated_autocorr.dat: header + '{n:4d} {mean:10.4f} {sem:10.4f}',
    sem = std(ddof=1)/sqrt(N) — byte-format-identical to aggregate_floquet.py."""
    d = tempfile.mkdtemp()
    series = [(0, [1.0, 0.5, 0.2]), (1, [1.0, 0.3, 0.1]), (2, [1.0, 0.4, 0.15])]
    mean, sem = aggregate_byo_autocorr(series, d)
    lines = open(os.path.join(d, "aggregated_autocorr.dat")).read().splitlines()
    assert lines[0] == "# kick  mean_autocorr  sem"
    # row 1: mean of [.5,.3,.4]=0.4; sem = std(ddof=1)/sqrt(3)
    arr = np.array([[0.5], [0.3], [0.4]])
    exp_sem = arr.std(ddof=1) / np.sqrt(3)
    assert abs(mean[1] - 0.4) < 1e-9
    assert abs(sem[1] - exp_sem) < 1e-9
    # np.loadtxt round-trip (what aggregate_floquet uses)
    agg = np.loadtxt(os.path.join(d, "aggregated_autocorr.dat"))
    assert np.allclose(agg[:, 1], mean)


def test_per_instance_dat_written_and_readable():
    d = tempfile.mkdtemp()
    aggregate_byo_autocorr([(0, [1.0, 0.5]), (1, [1.0, 0.4])], d)
    inst = np.loadtxt(os.path.join(d, "instance_00_autocorr.dat"))
    assert inst.shape == (2, 2)
    assert inst[0, 1] == 1.0 and inst[1, 1] == 0.5


def test_single_instance_sem_is_zero_not_nan():
    """One instance -> sem undefined; emit 0.0 so the .dat stays numeric."""
    d = tempfile.mkdtemp()
    _, sem = aggregate_byo_autocorr([(0, [1.0, 0.5, 0.2])], d)
    assert np.all(sem == 0.0)


# ------------------------- HDF5 group shape ------------------------------

def _byo_record(seed, env, source, phys, npi):
    return {
        "seed": seed, "script": "examples/byo/floquet_dtc.py",
        "placement_id": 0, "physical_qubit_set": phys, "env": env,
        "noise_source": source, "noise_placement_independent": npi,
        "num_kicks": [0, 1, 2], "autocorrelator": [1.0, 0.5, 0.2],
        "shots": 1000, "seed_simulator": 123456, "master_seed": 0,
    }


def test_write_byo_result_group_shape():
    """write_byo_result creates /byo/{stem}/seeds/seed_NNNN/placements/{phys}/{env}
    with the autocorrelator vector + provenance attrs — separate from the
    energy /devices tree."""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sweep.h5")
    rec = _byo_record(0, "device_calibrated", "device_calibrated",
                      ["QB1", "QB2", "QB3", "QB4"], True)
    with SweepHDF5Writer(path) as w:
        w.write_byo_result(rec)
    with h5py.File(path, "r") as f:
        grp = f["/byo/floquet_dtc/seeds/seed_0000/placements/QB1-QB2-QB3-QB4/device_calibrated"]
        assert list(grp["autocorrelator"][()]) == [1.0, 0.5, 0.2]
        assert list(grp["num_kicks"][()]) == [0, 1, 2]
        assert grp.attrs["noise_placement_independent"] == True   # noqa: E712
        assert grp.attrs["noise_source"] == "device_calibrated"
        assert grp.attrs["seed_simulator"] == 123456
        # RED-RESP-D3.4C §3: master_seed stored as the parent provenance knob;
        # 0 is a valid value and must be present (guards against a truthy check
        # that would drop master_seed=0, the committed gate-2 value).
        assert grp.attrs["master_seed"] == 0
        assert [q.decode() if isinstance(q, bytes) else q
                for q in grp["physical_qubit_set"][()]] == ["QB1", "QB2", "QB3", "QB4"]


def test_byo_and_energy_groups_coexist():
    """A BYO group and an energy SweepResultEntry write into the same file
    without collision (different top-level trees: /byo vs /devices)."""
    from lumi_hpc_qc.data.hdf5_writer import SweepResultEntry
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sweep.h5")
    entry = SweepResultEntry(
        device_id="q50", device_prefix="q50", seed=0,
        placement_qubits=["QB1"], calibration_id="cal0", noise_config="noiseless",
        energy_trajectory=[1.0, 0.5], best_energy=0.5, total_iterations=2,
        converged=True,
    )
    with SweepHDF5Writer(path) as w:
        w.write(entry)
        w.write_byo_result(_byo_record(0, "noiseless", "channels", ["QB1", "QB2"], False))
    with h5py.File(path, "r") as f:
        assert "/byo" in f and "/devices" in f


def test_noiseless_byo_record_flag_false():
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sweep.h5")
    with SweepHDF5Writer(path) as w:
        w.write_byo_result(_byo_record(1, "noiseless", "channels", ["QB1", "QB2"], False))
    with h5py.File(path, "r") as f:
        grp = f["/byo/floquet_dtc/seeds/seed_0001/placements/QB1-QB2/noiseless"]
        assert grp.attrs["noise_placement_independent"] == False  # noqa: E712


# --------------- WAL consistency + recovery (run-path) -------------------
# These cover the full run() self-check + crash-recovery, which the unit
# suite above does not: write_byo_result appends a WAL line AND creates an
# HDF5 group, so verify_consistency must count BYO groups and BYO WAL lines
# symmetrically, and recover_from_wal must be able to replay BYO rows.

def test_byo_run_consistency_no_false_positive():
    """A BYO-only run must NOT report WAL inconsistency. (Regression: BYO WAL
    lines used to lack group_path -> injected '' into wal_paths while no /byo
    group was counted -> consistent=False on every BYO run.)"""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sweep.h5")
    with SweepHDF5Writer(path) as w:
        w.write_byo_result(_byo_record(0, "device_calibrated", "device_calibrated",
                                       ["QB1", "QB2"], True))
        w.write_byo_result(_byo_record(1, "noiseless", "channels", ["QB1", "QB2"], False))
    report = SweepHDF5Writer(path).verify_consistency()
    assert report["consistent"] is True
    assert report["wal_entries"] == report["hdf5_groups"] == 2
    assert report["missing_from_hdf5"] == 0


def test_byo_and_energy_consistency():
    """Mixed BYO + energy run is consistent and counts both trees."""
    from lumi_hpc_qc.data.hdf5_writer import SweepResultEntry
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sweep.h5")
    entry = SweepResultEntry(
        device_id="q50", device_prefix="q50", seed=0,
        placement_qubits=["QB1"], calibration_id="cal0", noise_config="noiseless",
        energy_trajectory=[1.0, 0.5], best_energy=0.5, total_iterations=2,
        converged=True,
    )
    with SweepHDF5Writer(path) as w:
        w.write(entry)
        w.write_byo_result(_byo_record(0, "noiseless", "channels", ["QB1", "QB2"], False))
    report = SweepHDF5Writer(path).verify_consistency()
    assert report["consistent"] is True
    assert report["wal_entries"] == report["hdf5_groups"] == 2


def test_byo_wal_recovery_reconstructs_group():
    """If a BYO HDF5 group is lost but its WAL line survives, recover_from_wal
    rebuilds it byte-for-byte. (Regression: recovery used to skip BYO lines.)"""
    d = tempfile.mkdtemp()
    path = os.path.join(d, "sweep.h5")
    gpath = "/byo/floquet_dtc/seeds/seed_0000/placements/QB1-QB2/device_calibrated"
    with SweepHDF5Writer(path) as w:
        w.write_byo_result(_byo_record(0, "device_calibrated", "device_calibrated",
                                       ["QB1", "QB2"], True))
    # Simulate a group lost after the WAL fsync but before/around the HDF5 flush.
    with h5py.File(path, "a") as f:
        del f[gpath]
        assert gpath not in f
    recovered = SweepHDF5Writer(path).recover_from_wal()
    assert recovered == 1
    with h5py.File(path, "r") as f:
        assert gpath in f
        assert list(f[gpath]["autocorrelator"][()]) == [1.0, 0.5, 0.2]
        assert f[gpath].attrs["seed_simulator"] == 123456
        # master_seed (RED §3) must survive WAL replay too, not just the live write.
        assert f[gpath].attrs["master_seed"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
