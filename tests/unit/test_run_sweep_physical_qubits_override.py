# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""run_sweep CLI: the --physical-qubits override (PLACEMENT-1).

Verifies that `python3 -m lumi_hpc_qc.sweep.run_sweep <yaml> --physical-qubits Q,Q,...`:
  (1) injects ONE placement (a list of qubit names) onto every byo_circuit
      experiment in the same sub-dict parse_sweep_config reads from;
  (2) is SURGICAL — the only difference vs the no-flag config is that one key,
      so grid expansion / task_ids / the campaign manifest are unaffected
      (expand_grid derives the task set WITHOUT reading physical_qubits;
      placements multiply execution units inside _execute_byo_group, not the
      grid tasks). This is the "does not affect campaign logic" guarantee.
  (3) leaves non-byo experiments untouched;
  (4) errors (rc=1) if no byo_circuit experiment exists, or names parse empty;
  (5) never writes the YAML back to disk (in-memory override only).

Mirrors test_run_sweep_output_dir_override.py: run_sweep_from_dict is
monkeypatched to a capturing stub, so this asserts the dict the engine WOULD
receive without needing qiskit/Aer or running a sweep.
"""
from __future__ import annotations

import copy

import pytest
import yaml

import lumi_hpc_qc.sweep.run_sweep as run_sweep_cli

RUNNER = "QB1,QB2,QB5,QB6,QB7,QB9,QB10,QB11,QB12,QB13"
RUNNER_LIST = RUNNER.split(",")


def _write_yaml(tmp_path, *, byo=True, extra_nonbyo=False):
    """A minimal sweep YAML (top-level `sweep:` key) with selectable experiments."""
    exps = []
    if byo:
        exps.append({
            "type": "byo_circuit", "label": "t",
            "circuit_script": "examples/byo/floquet_dtc.py",
            "fixed": {"num_qubits": 10},
            "grid": {"num_kicks": {"range": [0, 2]}},
            "seed_list": [0], "shots": 8, "noise_configs": ["noiseless"],
        })
    if extra_nonbyo:
        exps.append({"type": "hamiltonian_battery", "label": "other", "model": "tfim"})
    p = tmp_path / "sweep.yaml"
    p.write_text(yaml.safe_dump({"sweep": {"experiments": exps}}), encoding="utf-8")
    return p


@pytest.fixture
def captured(monkeypatch):
    """Capture the config_dict run_sweep_from_dict is called with; don't run."""
    seen = {}

    def _stub(config_dict, *, device="CPU", progress_callback=None):
        seen["config_dict"] = copy.deepcopy(config_dict)
        seen["device"] = device
        return "SweepResult(stub)"

    monkeypatch.setattr(run_sweep_cli, "run_sweep_from_dict", _stub)
    return seen


def test_injects_single_placement_on_byo(tmp_path, captured):
    y = _write_yaml(tmp_path)
    rc = run_sweep_cli.main([str(y), "--physical-qubits", RUNNER])
    assert rc == 0
    exp = captured["config_dict"]["sweep"]["experiments"][0]
    assert exp["physical_qubits"] == RUNNER_LIST   # one placement, in order


def test_injection_is_surgical_task_set_unaffected(tmp_path, captured):
    """The override changes EXACTLY one key. Everything expand_grid reads (grid,
    seeds, fixed, ...) is byte-identical, so task_ids and the campaign manifest
    are unaffected — the campaign-logic guarantee."""
    y = _write_yaml(tmp_path)
    run_sweep_cli.main([str(y)])                    # no flag
    base = copy.deepcopy(captured["config_dict"])
    run_sweep_cli.main([str(y), "--physical-qubits", RUNNER])
    with_pq = copy.deepcopy(captured["config_dict"])
    # drop the one injected key -> must equal the no-flag config exactly
    with_pq["sweep"]["experiments"][0].pop("physical_qubits")
    assert with_pq == base


def test_nonbyo_experiment_untouched_and_whitespace_stripped(tmp_path, captured):
    y = _write_yaml(tmp_path, byo=True, extra_nonbyo=True)
    run_sweep_cli.main([str(y), "--physical-qubits", " QB1 , QB2 , QB5 "])
    exps = captured["config_dict"]["sweep"]["experiments"]
    byo = next(e for e in exps if e["type"] == "byo_circuit")
    other = next(e for e in exps if e["type"] != "byo_circuit")
    assert byo["physical_qubits"] == ["QB1", "QB2", "QB5"]   # stripped
    assert "physical_qubits" not in other


def test_error_when_no_byo_experiment(tmp_path):
    y = _write_yaml(tmp_path, byo=False, extra_nonbyo=True)
    assert run_sweep_cli.main([str(y), "--physical-qubits", RUNNER]) == 1


def test_error_when_names_parse_empty(tmp_path):
    y = _write_yaml(tmp_path)
    assert run_sweep_cli.main([str(y), "--physical-qubits", " , , "]) == 1


def test_no_flag_leaves_no_physical_qubits_key(tmp_path, captured):
    y = _write_yaml(tmp_path)
    run_sweep_cli.main([str(y)])
    assert "physical_qubits" not in captured["config_dict"]["sweep"]["experiments"][0]


def test_override_does_not_rewrite_yaml(tmp_path, captured):
    y = _write_yaml(tmp_path)
    before = y.read_text(encoding="utf-8")
    run_sweep_cli.main([str(y), "--physical-qubits", RUNNER])
    assert y.read_text(encoding="utf-8") == before
