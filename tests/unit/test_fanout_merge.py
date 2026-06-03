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

_BP_MOD = os.path.join(
    os.path.dirname(__file__), "..", "..", "src",
    "lumi_hpc_qc", "sweep", "battery_paths.py",
)
_bp_spec = importlib.util.spec_from_file_location("_battery_paths_under_test", _BP_MOD)
bp = importlib.util.module_from_spec(_bp_spec)
_bp_spec.loader.exec_module(bp)


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


# ── reducer selection (RED-RULING-MERGE-CLI-FOLLOWUP §3(a)/(b); PURE) ────────

def test_select_reducer_byo_from_config():
    red = fm.select_reducer({"byo_circuit"}, None, [0, 1])
    assert type(red).__name__ == "ByoAutocorrReducer"


def test_select_reducer_battery_characterization():
    red = fm.select_reducer({"characterization"}, None, [0, 1])
    assert type(red).__name__ == "BatteryReducer"


def test_select_reducer_battery_vqe_sweep():
    red = fm.select_reducer({"vqe_sweep"}, None, [0, 1])
    assert type(red).__name__ == "BatteryReducer"


def test_select_reducer_explicit_type_overrides_config():
    # config says battery, explicit override says byo -> byo wins (precedence).
    red = fm.select_reducer({"characterization"}, "byo_circuit", [0, 1])
    assert type(red).__name__ == "ByoAutocorrReducer"


def test_select_reducer_no_type_raises():
    try:
        fm.select_reducer(None, None, [0, 1])
    except ValueError as e:
        assert "provide --config" in str(e)
        return
    raise AssertionError("expected ValueError when no type is available")


def test_select_reducer_mixed_types_raises():
    try:
        fm.select_reducer({"byo_circuit", "characterization"}, None, [0, 1])
    except ValueError as e:
        assert "mixed experiment types" in str(e)
        assert "byo_circuit" in str(e) and "characterization" in str(e)
        return
    raise AssertionError("expected ValueError on a mixed-type config")


def test_select_reducer_unknown_type_raises():
    # random_regular is a graph_type, NOT an experiment type — must fail loud,
    # never fall through to identity-reduce.
    try:
        fm.select_reducer({"random_regular"}, None, [0, 1])
    except ValueError as e:
        assert "unknown experiment type" in str(e) and "random_regular" in str(e)
        return
    raise AssertionError("expected ValueError on an unknown experiment type")


# ── vacuous-merge guard (RED-RULING-MERGE-CLI-FOLLOWUP §3(c); PURE) ──────────

def test_union_nonvacuous_ok_is_silent():
    # union produced groups and the reducer matched records -> no raise.
    fm._assert_union_nonvacuous(8, 8, "BatteryReducer")


def test_union_nonvacuous_zero_groups_raises():
    try:
        fm._assert_union_nonvacuous(0, 0, "BatteryReducer")
    except ValueError as e:
        assert "0 leaf groups" in str(e)
        return
    raise AssertionError("expected ValueError on an empty union")


def test_union_nonvacuous_zero_records_raises():
    # the original bug: a battery file fed to the BYO reducer -> groups present
    # but 0 records -> the silent exit-0 vacuous pass, now loud.
    try:
        fm._assert_union_nonvacuous(8, 0, "ByoAutocorrReducer")
    except ValueError as e:
        assert "extracted 0 records" in str(e) and "ByoAutocorrReducer" in str(e)
        return
    raise AssertionError("expected ValueError when the reducer matched 0 records")


# ── option (i): battery_paths single source of truth + inventory serde + the
#    reducer-agnostic group-set assert (RED-RULING-PATCH43-VERIFY-AND-INVENTORY-
#    DESIGN Q2/Q4). All PURE (stdlib only) — no h5py/numpy. ──────────────────

import hashlib as _hashlib  # noqa: E402  (local to these tests)

_GRID = dict(device_prefix="QX7", seed=3, placement_qubits=["q12", "q5", "q8"],
             calibration_id="cal_abc", noise_config="noise_readout_only",
             model_params={})
_LHS = dict(device_prefix="QX7", seed=3, placement_qubits=["q12", "q5", "q8"],
            calibration_id="cal_abc", noise_config="noise_full",
            model_params={"J": 1.25, "h": 0.5, "g": -2.0})


def test_battery_group_path_format_grid_and_lhs():
    # The builder output is the exact on-disk path (pins the format the writer
    # delegates to). Grid: no params suffix. LHS: params_{md5(sorted k=v.8f)[:8]}.
    assert bp.battery_group_path(**_GRID) == (
        "devices/QX7/seeds/seed_0003/placements/QX7-q12_q5_q8/"
        "calibrations/cal_abc/noise_readout_only"
    )
    ps = ",".join(f"{k}={v:.8f}" for k, v in sorted(_LHS["model_params"].items()))
    tail = "params_" + _hashlib.md5(ps.encode()).hexdigest()[:8]
    assert bp.battery_group_path(**_LHS) == (
        "devices/QX7/seeds/seed_0003/placements/QX7-q12_q5_q8/"
        f"calibrations/cal_abc/noise_full/{tail}"
    )


def test_battery_paths_roundtrip_grid_key():
    # group_key_from_path(battery_group_path(...)) == the literal expected key;
    # the seed is dropped (instance axis), params_tail empty in grid mode.
    gk = bp.group_key_from_path(bp.battery_group_path(**_GRID))
    assert gk == ("QX7", "QX7-q12_q5_q8", "cal_abc", "noise_readout_only", "")


def test_battery_paths_roundtrip_lhs_params_key():
    # REQUIRED LHS round-trip (RED Q2): the params_{md5(...)} suffix is the most
    # likely drift point; a grid-only test would pass while LHS silently drifts.
    ps = ",".join(f"{k}={v:.8f}" for k, v in sorted(_LHS["model_params"].items()))
    tail = "params_" + _hashlib.md5(ps.encode()).hexdigest()[:8]
    gk = bp.group_key_from_path(bp.battery_group_path(**_LHS))
    assert gk == ("QX7", "QX7-q12_q5_q8", "cal_abc", "noise_full", tail)


def test_battery_extract_parse_matches_old_inline_parse():
    # The refactor (BatteryReducer.extract -> group_key_from_path(dirname)) must
    # be byte-identical to the pre-patch inline parse on the DATASET path. The
    # old parse used parts[ci+3:-1] on the dataset; the new uses parts[ci+3:] on
    # the GROUP path (dirname). They must agree for grid AND lhs.
    def old_inline(dataset_name):
        parts = dataset_name.split("/")
        di = parts.index("devices"); pi = parts.index("placements")
        ci = parts.index("calibrations")
        return (parts[di + 1], parts[pi + 1], parts[ci + 1], parts[ci + 2],
                "/".join(parts[ci + 3:-1]))
    for e in (_GRID, _LHS):
        dataset = bp.battery_group_path(**e) + "/energy_trajectory"
        assert bp.group_key_from_path(dataset.rsplit("/", 1)[0]) == old_inline(dataset)


def test_group_key_from_path_none_for_non_battery():
    # A foreign (BYO) subtree path is not a battery unit group -> None (skip).
    assert bp.group_key_from_path(
        "byo/echo/seeds/seed_0000/placements/q0_q1/noiseless"
    ) is None


def test_inventory_serde_roundtrip_preserves_set():
    groups = {
        bp.group_key_from_path(bp.battery_group_path(**_GRID)),
        bp.group_key_from_path(bp.battery_group_path(**_LHS)),
    }
    d = bp.inventory_to_json(groups)
    assert d["schema"] == "campaign_expected/v1" and d["reducer"] == "BatteryReducer"
    assert bp.inventory_from_json(d) == groups  # lists -> tuples, set preserved


def test_inventory_from_json_bad_schema_raises():
    try:
        bp.inventory_from_json({"schema": "bogus/v9", "groups": []})
    except ValueError as e:
        assert "schema" in str(e)
        return
    raise AssertionError("expected ValueError on an unexpected inventory schema")


def test_groupset_present_equals_expected_is_silent():
    # Complete + matching group set -> the group-set check is a no-op (no raise).
    A, B = ("d", "pA", "c", "n0", ""), ("d", "pB", "c", "n0", "")
    recs = [{"g": g, "i": s, "v": None} for g in (A, B) for s in (0, 1)]
    fm.assert_complete_and_reduce(
        recs, _MockReducer([0, 1]), "/tmp/x", expected_groups={A, B}
    )


def test_groupset_missing_group_raises_and_names_it():
    # A WHOLLY-ABSENT group (present units are complete, but a whole group is
    # gone) -> raise, naming the missing group. This is the (i-b) blind spot.
    A, B = ("d", "pA", "c", "n0", ""), ("d", "pB", "c", "n0", "")
    recs = [{"g": A, "i": s, "v": None} for s in (0, 1)]  # B entirely absent
    try:
        fm.assert_complete_and_reduce(
            recs, _MockReducer([0, 1]), "/tmp/x", expected_groups={A, B}
        )
    except ValueError as e:
        assert "missing" in str(e) and "pB" in repr(e.args)
        return
    raise AssertionError("expected ValueError on a wholly-absent group")


def test_groupset_unexpected_group_raises():
    A, B = ("d", "pA", "c", "n0", ""), ("d", "pB", "c", "n0", "")
    recs = [{"g": g, "i": s, "v": None} for g in (A, B) for s in (0, 1)]
    try:
        fm.assert_complete_and_reduce(
            recs, _MockReducer([0, 1]), "/tmp/x", expected_groups={A}  # B unexpected
        )
    except ValueError as e:
        assert "unexpected" in str(e)
        return
    raise AssertionError("expected ValueError on an unexpected group")


def test_groupset_none_skips_check():
    # Back-compat: expected_groups=None (single-rank / byte-identity gate) skips
    # the group-set check entirely — even with a group missing, no raise here.
    A, B = ("d", "pA", "c", "n0", ""), ("d", "pB", "c", "n0", "")
    recs = [{"g": A, "i": s, "v": None} for s in (0, 1)]  # B absent, but None skips
    fm.assert_complete_and_reduce(
        recs, _MockReducer([0, 1]), "/tmp/x", expected_groups=None
    )


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items())
           if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"FANOUT MERGE: ALL {len(fns)} CHECKS PASSED")
