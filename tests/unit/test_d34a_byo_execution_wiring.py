# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""D3.4a — BYO execution branch: grouping + dispatch + placement-from-connectivity.

Verifies the routing/placement seam, NOT the counts run (stubbed in D3.4a,
landed in D3.4b):
  - _group_tasks keys byo_circuit tasks by (circuit_script, "byo", cal, params),
    keeping the 4-tuple shape run() unpacks; they don't collapse with hamiltonian
    tasks.
  - _execute_group dispatches byo_circuit groups to _execute_byo_group.
  - _execute_byo_group builds the circuit, solves placements from the circuit's
    own connectivity (top_1 under the device_calibrated guardrail), and resolves
    noise_placement_independent + physical_qubit_set before the (stubbed) run.

The grouping test is pure. The dispatch/placement test needs qiskit (build +
solver) and runs in-container; it asserts the D3.4b stub raises with the wired
placement info, proving build+placement happened.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("qiskit")

from lumi_hpc_qc.sweep.sweep_engine import SweepEngine, SweepTask
from lumi_hpc_qc.sweep.noise_configs import NOISE_ENV_BY_NAME

_REPO = Path(__file__).resolve().parents[2]
_FACTORY = str(_REPO / "examples" / "byo" / "floquet_dtc.py")
_DISORDER = {
    "hz_angles": [0.86, -1.45, -2.88, -3.04],
    "Jzz_angles": [-2.16, -1.84, -2.81, -2.42],
    "init_bit_array": [0, 0, 0, 0],
}
_FIXED = {"num_qubits": 4, "epsilon": 0.03}
_CAL = str(_REPO / "examples" / "q50_calibration_20260524_08c3c70f.json")


def _byo_task(seed: int, num_kicks: int, noise_names: list[str]) -> SweepTask:
    return SweepTask(
        experiment_type="byo_circuit",
        circuit_script=_FACTORY,
        circuit_function="build_circuit",
        fixed_params=dict(_FIXED),
        disorder_instance=dict(_DISORDER),
        circuit_params={"num_kicks": num_kicks},
        disorder_gates=("rz", "rzz"),
        seed=seed,
        calibration_path=_CAL,
        max_placements=2,   # bound the noiseless-only placement count in tests
        noise_configs=[NOISE_ENV_BY_NAME[n] for n in noise_names],
    )


def _ham_task() -> SweepTask:
    return SweepTask(
        experiment_type="characterization",
        hamiltonian="tfim",
        topology_name="4q_chain",
        calibration_path=_CAL,
        noise_configs=[NOISE_ENV_BY_NAME["noiseless"]],
    )


# ----------------------------- grouping ----------------------------------

def test_byo_tasks_group_by_script_not_with_hamiltonian():
    eng = SweepEngine.__new__(SweepEngine)  # no full init; _group_tasks is pure
    tasks = [
        _byo_task(0, 5, ["noiseless"]),
        _byo_task(1, 5, ["noiseless"]),
        _ham_task(),
    ]
    groups = SweepEngine._group_tasks(eng, tasks)
    # two distinct groups: one BYO (2 seeds), one hamiltonian
    keys = list(groups.keys())
    byo_keys = [k for k in keys if k[1] == "byo"]
    ham_keys = [k for k in keys if k[1] != "byo"]
    assert len(byo_keys) == 1, keys
    assert len(ham_keys) == 1, keys
    assert len(groups[byo_keys[0]]) == 2          # both BYO seeds together
    assert byo_keys[0][0] == _FACTORY             # script in the ham slot
    # 4-tuple shape preserved (run() unpacks 4)
    assert all(len(k) == 4 for k in keys)


def test_byo_groups_split_by_script():
    eng = SweepEngine.__new__(SweepEngine)
    t1 = _byo_task(0, 5, ["noiseless"])
    t2 = _byo_task(0, 5, ["noiseless"])
    t2.circuit_script = "examples/byo/other_factory.py"
    groups = SweepEngine._group_tasks(eng, [t1, t2])
    assert len(groups) == 2


# ------------------ dispatch + placement (needs solver) ------------------

def _full_engine_for(tasks):
    """Build a SweepEngine wired enough to run _execute_byo_group via the real
    calibration contract: the engine's own _load_calibration populates
    _cal_cache and registers the device with the solver. Avoids run()."""
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
    from lumi_hpc_qc.plugins.registry import PluginRegistry
    eng = SweepEngine.__new__(SweepEngine)
    eng._solver = GeneralPlacementSolver()
    eng._registry = PluginRegistry()
    eng._registry.discover()
    eng._cal_cache = {}
    eng._timing = {"circuit_build_s": 0.0, "placement_solving_s": 0.0}
    eng._load_calibration(_CAL)   # real contract: caches + registers device
    return eng


def _seed_grid_tasks(seed: int, kicks: list[int], noise_names: list[str],
                     master_seed: int | None = 0) -> list[SweepTask]:
    """A seed's worth of BYO tasks across a num_kicks grid (one per kick),
    sharing one disorder instance — mirrors how expansion attaches them."""
    out = []
    for k in kicks:
        t = _byo_task(seed, k, noise_names)
        t.master_seed = master_seed
        out.append(t)
    return out


def test_byo_execute_computes_autocorrelator_device_cal():
    """D3.4b: _execute_byo_group builds the seed's kick-grid, runs batched under
    device_calibrated (top_1 + guardrail), and computes an autocorrelator series
    per (seed, placement, env)."""
    tasks = _seed_grid_tasks(0, [0, 1, 2, 3], ["device_calibrated", "noiseless"])
    eng = _full_engine_for(tasks)
    errors: list[str] = []
    eng._execute_byo_group(tasks, writer=None, errors=errors)
    assert errors == []
    res = eng._byo_results_last
    # one record per (seed=1) x (placement=1, top_1) x (env in {device_cal, noiseless})
    assert len(res) == 2
    for r in res:
        assert r["seed"] == 0
        assert len(r["autocorrelator"]) == 4          # one per kick 0..3
        assert r["num_kicks"] == [0, 1, 2, 3]         # ascending grid order
        assert len(r["physical_qubit_set"]) == 4      # q4 placement
        # device_calibrated -> guardrail on; noiseless -> off
        if r["noise_source"] == "device_calibrated":
            assert r["noise_placement_independent"] is True
        else:
            assert r["noise_placement_independent"] is False
    # num_kicks=0 autocorrelator is the t=0 reference: polarized init, all-zero
    # bitstring dominates -> A(0) ~ +1.0 (within shot noise).
    dc = next(r for r in res if r["noise_source"] == "device_calibrated")
    assert dc["autocorrelator"][0] > 0.8              # A(0) near +1


def test_byo_noiseless_only_no_guardrail():
    """A noiseless-only BYO group runs all placements (no device-cal guardrail)
    and does not stamp noise_placement_independent."""
    tasks = _seed_grid_tasks(0, [0, 1, 2], ["noiseless"])
    eng = _full_engine_for(tasks)
    errors: list[str] = []
    eng._execute_byo_group(tasks, writer=None, errors=errors)
    assert errors == []
    res = eng._byo_results_last
    assert all(r["noise_placement_independent"] is False for r in res)
    assert all(r["noise_source"] == "channels" for r in res)  # noiseless is channels-source


def test_byo_seed_simulator_is_per_instance_derived():
    """The seed_simulator used is resolve_instance_seed(master_seed, seed) —
    the bank's per-instance derivation, not the raw seed."""
    from lumi_hpc_qc.sweep.byo_observable import resolve_instance_seed
    tasks = _seed_grid_tasks(2, [0, 1], ["noiseless"], master_seed=0)
    eng = _full_engine_for(tasks)
    eng._execute_byo_group(tasks, writer=None, errors=[])
    res = eng._byo_results_last
    expected = resolve_instance_seed(0, 2)
    assert all(r["seed_simulator"] == expected for r in res)
    assert expected != 2                              # derived, not the raw seed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
