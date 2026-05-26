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


def test_byo_execute_builds_and_places_then_stubs():
    """_execute_byo_group builds the circuit, solves top_1 placement from
    connectivity, resolves the guardrail, then raises the D3.4b stub carrying
    the wired physical_qubit_set + noise_placement_independent."""
    tasks = [_byo_task(0, 5, ["device_calibrated", "noiseless"])]
    eng = _full_engine_for(tasks)
    errors: list[str] = []
    with pytest.raises(NotImplementedError) as ei:
        eng._execute_byo_group(tasks, writer=None, errors=errors)
    msg = str(ei.value)
    assert "D3.4b" in msg
    assert "noise_placement_independent=True" in msg     # device_calibrated -> guardrail on
    assert "physical_qubit_set=" in msg                  # placement resolved
    assert errors == []                                  # build+place succeeded


def test_byo_noiseless_only_no_placement_independent_flag():
    """A noiseless-only BYO group does not set the device-cal guardrail."""
    tasks = [_byo_task(0, 5, ["noiseless"])]
    eng = _full_engine_for(tasks)
    with pytest.raises(NotImplementedError) as ei:
        eng._execute_byo_group(tasks, writer=None, errors=[])
    assert "noise_placement_independent=False" in str(ei.value)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
