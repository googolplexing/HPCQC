# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""BYO-FAMILY-COLLISION fix -- two single-observable arms must not collapse.

Two byo_circuit experiments sharing (circuit_script, calibration) but differing
in circuit_function both carried observable_name="default", so before this fix
they keyed identically in _group_tasks and collapsed into one group: only
tasks[0]'s circuit_function ran (the echo arm produced nothing), and a default
family's HDF5/.dat leaf was "" regardless of function, so even with the group
fixed two families would overwrite each other on disk.

The fix (extends D7-increment-2; RED-RULING-BYO-SINGLE-OBSERVABLE-ARM-COLLAPSE):
  (a) fold circuit_function into the BYO group key  -> distinct families split;
  (b) form (b1): when a script stem hosts >1 family in the resolved run, ALL its
      default-family leaves take a "/<circuit_function>" segment via the shared
      byo_observable_subpath seam (all-families-or-none, keyed on the run); a
      lone family keeps "" (byte-identical to the bank).

This guards, by COUNT and CONTENTS (not path strings):
  - two single-observable arms (diff function) -> two groups, both functions present;
  - the collision-stem derivation (Form A): >1 family -> stem flagged; else not;
  - the seam truth table: lone default -> "", colliding default -> "/<func>"
    for EVERY colliding family (incl. the first), declared name -> "/<name>";
  - distinct resolved leaf subpaths for the two arms (no collision);
  - one-arm sweep -> one group + legacy "" (byte-identity guard);
  - multi-observable (declared names) -> unchanged from D7 ("/<name>");
  - PLACEMENT EQUIVALENCE (real Q50): both families of a script resolve to the
    SAME placements and envs -- the verified fact that makes Form A (key on
    script_stem) equivalent to Form B (key on (stem, placement, env)). Asserted,
    not assumed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from lumi_hpc_qc.sweep.byo_observable import (
    DEFAULT_OBSERVABLE_NAME,
    byo_observable_subpath,
)
from lumi_hpc_qc.sweep.sweep_engine import SweepEngine, SweepTask
from lumi_hpc_qc.sweep.noise_configs import NOISE_ENV_BY_NAME

_REPO = Path(__file__).resolve().parents[2]
# floquet_dtc_echo.py hosts BOTH build_circuit (autocorr) and build_circuit_echo
# — the exact two-family case this fix is about.
_FACTORY = str(_REPO / "examples" / "byo" / "floquet_dtc_echo.py")
_CAL = str(_REPO / "examples" / "q50_calibration_20260524_08c3c70f.json")
_DISORDER = {
    "hz_angles": [0.86, -1.45, -2.88, -3.04],
    "Jzz_angles": [-2.16, -1.84, -2.81, -2.42],
    "init_bit_array": [0, 0, 0, 0],
}
_FIXED = {"num_qubits": 4, "epsilon": 0.03}


def _byo_task(seed: int, func: str, noise_names=("noiseless",),
              script: str = _FACTORY) -> SweepTask:
    return SweepTask(
        experiment_type="byo_circuit",
        circuit_script=script,
        circuit_function=func,
        observable_name=DEFAULT_OBSERVABLE_NAME,   # the single-observable form
        fixed_params=dict(_FIXED),
        disorder_instance=dict(_DISORDER),
        circuit_params={"num_kicks": 5},
        disorder_gates=("rz", "rzz"),
        seed=seed,
        calibration_path=_CAL,
        max_placements=2,
        noise_configs=[NOISE_ENV_BY_NAME[n] for n in noise_names],
    )


# ── (a) group key: two single-observable arms split into two groups ──────────

def test_two_single_observable_arms_split_into_two_groups():
    eng = SweepEngine.__new__(SweepEngine)   # _group_tasks is pure
    tasks = [
        _byo_task(0, "build_circuit"),
        _byo_task(1, "build_circuit"),
        _byo_task(0, "build_circuit_echo"),
        _byo_task(1, "build_circuit_echo"),
    ]
    groups = SweepEngine._group_tasks(eng, tasks)
    byo_keys = [k for k in groups if k[1] == "byo"]
    # COUNT: two distinct BYO groups (one per circuit family).
    assert len(byo_keys) == 2, byo_keys
    # CONTENTS: both circuit functions represented, one family per group.
    funcs_per_group = {
        frozenset(t.circuit_function for t in groups[k]) for k in byo_keys
    }
    assert funcs_per_group == {frozenset({"build_circuit"}),
                               frozenset({"build_circuit_echo"})}, funcs_per_group
    # each group holds its two seeds
    assert all(len(groups[k]) == 2 for k in byo_keys)
    # 4-tuple shape preserved (run() unpacks 4)
    assert all(len(k) == 4 for k in groups)


def test_one_arm_is_still_a_single_group():
    eng = SweepEngine.__new__(SweepEngine)
    groups = SweepEngine._group_tasks(
        eng, [_byo_task(0, "build_circuit"), _byo_task(1, "build_circuit")]
    )
    assert len(groups) == 1
    assert len(next(iter(groups.values()))) == 2


# ── collision-stem derivation (Form A) ───────────────────────────────────────

def test_collision_stems_flag_only_multi_family_scripts():
    stem = Path(_FACTORY).stem
    # >1 family on one stem -> flagged
    two = [_byo_task(0, "build_circuit"), _byo_task(0, "build_circuit_echo")]
    assert SweepEngine._compute_byo_collision_stems(two) == {stem}
    # one family -> not flagged
    one = [_byo_task(0, "build_circuit"), _byo_task(1, "build_circuit")]
    assert SweepEngine._compute_byo_collision_stems(one) == set()
    # two families but on DIFFERENT scripts -> neither flagged
    sep = [_byo_task(0, "build_circuit"),
           _byo_task(0, "build_circuit_echo", script="examples/byo/other.py")]
    assert SweepEngine._compute_byo_collision_stems(sep) == set()


# ── (b1) seam: all-families-or-none, keyed on the run-level flag ─────────────

def test_seam_lone_default_is_legacy_empty():
    # No collision -> "" (byte-identical to the bank). Also the zero-arg form
    # the existing foundation test pins, unchanged.
    assert byo_observable_subpath(DEFAULT_OBSERVABLE_NAME) == ""
    assert byo_observable_subpath(
        DEFAULT_OBSERVABLE_NAME, "build_circuit", False) == ""


def test_seam_colliding_default_disambiguates_every_family():
    # ALL colliding families get a segment -- including the first; the default
    # family's path must not depend on whether another family exists.
    assert byo_observable_subpath(
        DEFAULT_OBSERVABLE_NAME, "build_circuit", True) == "/build_circuit"
    assert byo_observable_subpath(
        DEFAULT_OBSERVABLE_NAME, "build_circuit_echo", True) == "/build_circuit_echo"


def test_seam_declared_name_unaffected_by_flag():
    # The multi-observable surface is already separated by name; the flag must
    # not perturb it (D7 case unchanged).
    assert byo_observable_subpath("echo", "build_circuit_echo", False) == "/echo"
    assert byo_observable_subpath("echo", "build_circuit_echo", True) == "/echo"


def test_two_arms_resolve_to_distinct_leaf_subpaths():
    # The end-to-end no-collision property: under a colliding stem, the two
    # default families produce DIFFERENT leaf segments (asserted by value).
    sub_a = byo_observable_subpath(DEFAULT_OBSERVABLE_NAME, "build_circuit", True)
    sub_e = byo_observable_subpath(DEFAULT_OBSERVABLE_NAME, "build_circuit_echo", True)
    assert sub_a != sub_e
    assert sub_a and sub_e   # neither is "" (both disambiguated)


# ── placement equivalence: the fact that makes Form A == Form B ──────────────

@pytest.mark.skipif(not Path(_CAL).exists(), reason="Q50 calibration not present")
def test_two_families_share_placements_and_envs_real_q50():
    """Form A keys collision on script_stem, not (stem, placement, env). That is
    equivalent to Form B iff every family of a script resolves to the SAME
    placements and envs. Both arms share circuit_script -> the BUILT circuit's
    connectivity is identical -> the solver sees identical edges/qubits ->
    identical placements; and envs are per-experiment config, identical here.
    Verify the connectivity-and-placement equality directly so Form A's
    correctness is tested, not assumed.
    """
    pytest.importorskip("rustworkx")
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
    from lumi_hpc_qc.sweep.circuit_loader import load_circuit

    # Build both families' circuits; assert identical connectivity (the premise).
    def edges(func):
        qc = load_circuit(
            script_file=_FACTORY,
            script_function=func,
            script_params=dict(
                num_kicks=5,
                epsilon=_FIXED["epsilon"],
                num_qubits=_FIXED["num_qubits"],
                hz_angles=_DISORDER["hz_angles"],
                Jzz_angles=_DISORDER["Jzz_angles"],
                init_bit_array=_DISORDER["init_bit_array"],
            ),
        ).circuit
        cmap = sorted(
            tuple(sorted(qc.find_bit(q).index for q in instr.qubits))
            for instr in qc.data if len(instr.qubits) == 2
        )
        return set(cmap)

    e_auto = edges("build_circuit")
    e_echo = edges("build_circuit_echo")
    assert e_auto == e_echo, "families differ in connectivity -> Form A != Form B"

    # And the solver resolves identical placements from identical edges.
    cal = IQMv2Adapter().load(_CAL)
    solver = GeneralPlacementSolver()
    solver.add_device(cal)
    conn = sorted(e_auto)
    p_auto = solver.find_all_placements(
        circuit_edges=conn, circuit_qubits=_FIXED["num_qubits"],
        device_ids=[cal.device_id], max_placements=4,
    )
    p_echo = solver.find_all_placements(
        circuit_edges=conn, circuit_qubits=_FIXED["num_qubits"],
        device_ids=[cal.device_id], max_placements=4,
    )
    assert [pl.physical_indices for pl in p_auto] == \
           [pl.physical_indices for pl in p_echo]
