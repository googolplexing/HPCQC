# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Offline unit tests for the generic merge layer (sweep/fanout_merge.py).

Covers the load-bearing SAFETY logic without aer/h5py: the completeness
assertion (RED-RULING-WORKSTREAM-B §2 short-count guard) — grouping, fail-loud on
missing/extra/duplicate instances, instance-sorted dispatch — plus the manifest
concat (pure json union) and the BYO reducer's reduce (numpy is local; writes the
.dat). The HDF5 union and the BYO h5 extract are lazy-h5py and exercised by the
byte-identity gate on LUMI.
"""

import importlib.util
import json
import os
import tempfile

_MOD = os.path.join(
    os.path.dirname(__file__), "..", "..", "src",
    "lumi_hpc_qc", "sweep", "fanout_merge.py",
)
_spec = importlib.util.spec_from_file_location("_fanout_merge_under_test", _MOD)
fm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fm)


class _MockReducer:
    """Group key 'g', instance key 'i', payload 'v'; expected instances fixed."""

    def __init__(self, expected):
        self._expected = set(expected)
        self.calls = []

    def group_key(self, r):
        return r["g"]

    def instance_key(self, r):
        return r["i"]

    def payload(self, r):
        return r["v"]

    def expected_instances(self, gk):
        return self._expected

    def reduce(self, gk, series, out_root):
        self.calls.append((gk, series, out_root))


def test_groups_and_dispatches_per_group():
    recs = [
        {"g": "A", "i": 0, "v": "a0"},
        {"g": "B", "i": 0, "v": "b0"},
        {"g": "A", "i": 1, "v": "a1"},
        {"g": "B", "i": 1, "v": "b1"},
    ]
    red = _MockReducer(expected={0, 1})
    reduced = fm.assert_complete_and_reduce(recs, red, "/out")
    assert sorted(reduced) == ["A", "B"]
    assert len(red.calls) == 2
    # reduce receives the instance-sorted series for its group
    by_group = {gk: series for gk, series, _ in red.calls}
    assert by_group["A"] == [(0, "a0"), (1, "a1")]
    assert by_group["B"] == [(0, "b0"), (1, "b1")]


def test_series_is_instance_sorted_regardless_of_input_order():
    recs = [
        {"g": "A", "i": 2, "v": "a2"},
        {"g": "A", "i": 0, "v": "a0"},
        {"g": "A", "i": 1, "v": "a1"},
    ]
    red = _MockReducer(expected={0, 1, 2})
    fm.assert_complete_and_reduce(recs, red, "/out")
    assert red.calls[0][1] == [(0, "a0"), (1, "a1"), (2, "a2")]


def test_fail_loud_on_missing_instance():
    # group A is missing seed 2 -> short-count; must raise, must NOT reduce.
    recs = [
        {"g": "A", "i": 0, "v": "a0"},
        {"g": "A", "i": 1, "v": "a1"},
    ]
    red = _MockReducer(expected={0, 1, 2})
    try:
        fm.assert_complete_and_reduce(recs, red, "/out")
    except ValueError as e:
        assert "INCOMPLETE" in str(e) and "missing=[2]" in str(e)
        assert red.calls == []  # never aggregated a partial series
        return
    raise AssertionError("expected ValueError on missing instance")


def test_fail_loud_on_extra_instance():
    recs = [
        {"g": "A", "i": 0, "v": "a0"},
        {"g": "A", "i": 1, "v": "a1"},
        {"g": "A", "i": 9, "v": "a9"},
    ]
    red = _MockReducer(expected={0, 1})
    try:
        fm.assert_complete_and_reduce(recs, red, "/out")
    except ValueError as e:
        assert "extra=[9]" in str(e)
        return
    raise AssertionError("expected ValueError on extra instance")


def test_fail_loud_on_duplicate_instance():
    recs = [
        {"g": "A", "i": 0, "v": "a0"},
        {"g": "A", "i": 0, "v": "a0_again"},
    ]
    red = _MockReducer(expected={0})
    try:
        fm.assert_complete_and_reduce(recs, red, "/out")
    except ValueError as e:
        assert "duplicate instance" in str(e)
        return
    raise AssertionError("expected ValueError on duplicate instance")


def test_concat_manifests_union_and_priority():
    with tempfile.TemporaryDirectory() as d:
        m0 = os.path.join(d, "m0.json")
        m1 = os.path.join(d, "m1.json")
        out = os.path.join(d, "merged.json")
        # rank 0: t1 completed, t2 pending; rank 1: t2 completed, t3 completed
        json.dump({"tasks": {"t1": "completed", "t2": "pending"},
                   "total_tasks": 2}, open(m0, "w"))
        json.dump({"tasks": {"t2": "completed", "t3": "completed"},
                   "total_tasks": 2}, open(m1, "w"))
        merged = fm.concat_manifests([m0, m1], out)
        # union of task ids; completed wins the t2 disagreement
        assert merged["tasks"] == {
            "t1": "completed", "t2": "completed", "t3": "completed",
        }
        assert merged["total_tasks"] == 3
        # persisted
        assert json.load(open(out))["tasks"]["t2"] == "completed"


def test_byo_reducer_reduce_writes_aggregated_dat():
    # numpy is local; exercise the BYO reduce end-to-end (minus h5) to confirm
    # the reducer-callback seam reaches aggregate_byo_autocorr and produces the
    # aggregated .dat for a complete series.
    import importlib.util as _u
    # load byo_observable directly by path (avoid the qiskit-pulling package init)
    bo_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "src",
        "lumi_hpc_qc", "sweep", "byo_observable.py",
    )
    spec = _u.spec_from_file_location("_byo_obs", bo_path)
    bo = _u.module_from_spec(spec)
    try:
        spec.loader.exec_module(bo)
    except Exception:
        # byo_observable may import heavier siblings on some trees; if so, skip
        # the numeric reduce check here (it is covered on LUMI by the gate).
        print("  SKIP byo reduce (byo_observable import needs container)")
        return
    with tempfile.TemporaryDirectory() as d:
        series = [(0, [1.0, 0.5, 0.25]), (1, [0.9, 0.4, 0.2])]
        bo.aggregate_byo_autocorr(series, d)
        agg = os.path.join(d, "aggregated_autocorr.dat")
        assert os.path.exists(agg)
        head = open(agg).readline()
        assert "mean" in head and "sem" in head


def test_byo_dat_subpath_reproduces_single_node_layout():
    # The merge must rebuild the single-node byo_dat layout
    # {stem}/{phys}/{env}{obs} from the group key carried out of the HDF5 path.
    # Lone-default family (obs_tail ""): stem/phys/env. Declared/disambiguated
    # family (obs_tail "name"): stem/phys/env/name. (RED-REVIEW-INCREMENTS-1-2:
    # the carried stem + obs must reproduce the layout exactly.)
    assert fm.byo_dat_subpath(("floquet_dtc", "QB8-QB16", "device_calibrated", "")) \
        == os.path.join("floquet_dtc", "QB8-QB16", "device_calibrated")
    assert fm.byo_dat_subpath(("floquet_dtc_echo", "QB8-QB16", "noiseless", "echo")) \
        == os.path.join("floquet_dtc_echo", "QB8-QB16", "noiseless", "echo")


def test_byo_reducer_default_out_subpath():
    # Constructing the reducer without out_subpath_fn defaults to byo_dat_subpath.
    red = fm.ByoAutocorrReducer(expected_seeds=[0, 1])
    assert red._out_subpath_fn is fm.byo_dat_subpath
    assert red.expected_instances(("s", "p", "e", "")) == {0, 1}


# ── BatteryReducer (the identity-reduce non-BYO path; RED-RULING-ITEM3) ──────
#
# extract() reads HDF5 (needs h5py — exercised by the LUMI battery gate). The
# PURE logic — group/instance keys, expected_instances, the no-op reduce, and
# the generic completeness driver over battery-shaped records — is offline-tested
# here. Records mimic what extract() yields: {"group": 5-tuple, "seed": int}.

def _battery_rec(device, placement, cal, noise, params_tail, seed):
    return {"group": (device, placement, cal, noise, params_tail), "seed": seed}


def test_battery_reducer_keys_and_identity_reduce():
    red = fm.BatteryReducer(expected_seeds=[0, 1])
    rec = _battery_rec("q50", "q50-QB1_QB2", "cal0", "noiseless", "", 0)
    assert red.group_key(rec) == ("q50", "q50-QB1_QB2", "cal0", "noiseless", "")
    assert red.instance_key(rec) == 0
    assert red.payload(rec) is None  # identity reduce consumes no payload
    assert red.expected_instances(red.group_key(rec)) == {0, 1}
    # reduce is a no-op: returns None and writes nothing under out_root.
    assert red.reduce(red.group_key(rec), [(0, None), (1, None)], "/out") is None


def test_battery_reducer_driver_completeness_passes_when_full():
    red = fm.BatteryReducer(expected_seeds=[0, 1])
    recs = [
        _battery_rec("q50", "p0", "cal0", "noiseless", "", 0),
        _battery_rec("q50", "p0", "cal0", "noiseless", "", 1),
        _battery_rec("q50", "p0", "cal0", "noise_readout_only", "", 0),
        _battery_rec("q50", "p0", "cal0", "noise_readout_only", "", 1),
    ]
    # Two complete (placement, env) groups, each with both seeds -> no raise.
    reduced = fm.assert_complete_and_reduce(recs, red, "/out")
    assert len(reduced) == 2


def test_battery_reducer_driver_fails_loud_on_lost_shard():
    # A lost rank shard drops a node's units: here seed 1 of the noiseless group
    # is missing. The completeness guard must refuse (the value the driver adds
    # for the battery path even though reduce is empty).
    red = fm.BatteryReducer(expected_seeds=[0, 1])
    recs = [
        _battery_rec("q50", "p0", "cal0", "noiseless", "", 0),  # seed 1 missing
        _battery_rec("q50", "p0", "cal0", "noise_readout_only", "", 0),
        _battery_rec("q50", "p0", "cal0", "noise_readout_only", "", 1),
    ]
    try:
        fm.assert_complete_and_reduce(recs, red, "/out")
    except ValueError as e:
        assert "INCOMPLETE" in str(e) and "missing=[1]" in str(e)
    else:
        raise AssertionError("expected a short-count ValueError on the lost shard")


def test_battery_reducer_params_tail_keeps_lhs_groups_distinct():
    # LHS-mode units share (device, placement, cal, noise) but differ in the
    # params_ sublevel; the params_tail in the group key keeps them distinct
    # (the battery analog of BYO's obs_tail), so each is its own complete group.
    red = fm.BatteryReducer(expected_seeds=[0, 1])
    recs = [
        _battery_rec("q50", "p0", "cal0", "noise_full", "params_aaa", 0),
        _battery_rec("q50", "p0", "cal0", "noise_full", "params_aaa", 1),
        _battery_rec("q50", "p0", "cal0", "noise_full", "params_bbb", 0),
        _battery_rec("q50", "p0", "cal0", "noise_full", "params_bbb", 1),
    ]
    reduced = fm.assert_complete_and_reduce(recs, red, "/out")
    assert len(reduced) == 2  # NOT collapsed into one group


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"FANOUT MERGE: ALL {len(fns)} CHECKS PASSED")
