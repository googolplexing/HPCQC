# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""run_sweep CLI: the --output-dir override.

Verifies that `python3 -m lumi_hpc_qc.sweep.run_sweep <yaml> --output-dir DIR`:
  (1) injects DIR into the same sub-dict parse_sweep_config reads from
      (`config_dict["sweep"]["output_dir"]`), so the engine writes there;
  (2) leaves the config's output_dir untouched when the flag is absent;
  (3) never writes the YAML file back to disk (in-memory override only).

Motivation: the W1.6 gate runs the canonical, unmodified
examples/byo/floquet_dtc_q10_sweep.yaml into a clean run-specific workdir
(sweep_output/w1_gate) without duplicating the config or editing it. The
override is what lets one canonical config feed both the production sweep and
the gate, so there is no drift surface between them.

The CLI is exercised with run_sweep_from_dict monkeypatched to a capturing
stub, so this test asserts the dict the engine WOULD receive without needing
qiskit/Aer or running a sweep.
"""

from __future__ import annotations

import textwrap

import pytest

import lumi_hpc_qc.sweep.run_sweep as run_sweep_cli


def _write_yaml(tmp_path, *, with_output_dir):
    """A minimal valid sweep YAML (top-level `sweep:` key)."""
    od = "  output_dir: sweep_output/from_yaml\n" if with_output_dir else ""
    text = textwrap.dedent(
        """\
        sweep:
        __OUTPUT_DIR__  experiments:
            - type: byo_circuit
              label: t
              circuit_script: examples/byo/floquet_dtc.py
              circuit_function: build_circuit
              fixed: {num_qubits: 10}
              grid: {num_kicks: {range: [0, 2]}}
              seed_list: [0]
              shots: 8
              noise_configs: [noiseless]
        """
    ).replace("__OUTPUT_DIR__", od.rstrip("\n"))
    p = tmp_path / "sweep.yaml"
    p.write_text(text, encoding="utf-8")
    return p


@pytest.fixture
def captured(monkeypatch):
    """Capture the config_dict run_sweep_from_dict is called with; don't run."""
    seen = {}

    class _Result:
        def __repr__(self):
            return "SweepResult(stub)"

    def _stub(config_dict, *, device="CPU", progress_callback=None):
        seen["config_dict"] = config_dict
        seen["device"] = device
        return _Result()

    monkeypatch.setattr(run_sweep_cli, "run_sweep_from_dict", _stub)
    return seen


def test_output_dir_override_injects_into_sweep_section(tmp_path, captured):
    yaml_path = _write_yaml(tmp_path, with_output_dir=True)
    rc = run_sweep_cli.main([str(yaml_path), "--output-dir", "sweep_output/w1_gate"])
    assert rc == 0
    cfg = captured["config_dict"]
    assert cfg["sweep"]["output_dir"] == "sweep_output/w1_gate"


def test_override_replaces_yaml_value(tmp_path, captured):
    """An explicit YAML output_dir is overridden, not appended-to."""
    yaml_path = _write_yaml(tmp_path, with_output_dir=True)
    run_sweep_cli.main([str(yaml_path), "--output-dir", "sweep_output/w1_gate"])
    assert captured["config_dict"]["sweep"]["output_dir"] == "sweep_output/w1_gate"


def test_no_flag_leaves_yaml_output_dir_untouched(tmp_path, captured):
    yaml_path = _write_yaml(tmp_path, with_output_dir=True)
    run_sweep_cli.main([str(yaml_path)])
    assert captured["config_dict"]["sweep"]["output_dir"] == "sweep_output/from_yaml"


def test_no_flag_and_no_yaml_value_means_no_key(tmp_path, captured):
    """Absent flag + absent YAML field -> no output_dir key injected
    (the engine's own default of "sweep_output" then applies downstream)."""
    yaml_path = _write_yaml(tmp_path, with_output_dir=False)
    run_sweep_cli.main([str(yaml_path)])
    assert "output_dir" not in captured["config_dict"]["sweep"]


def test_override_does_not_rewrite_yaml_file(tmp_path, captured):
    """The override is in-memory; the file on disk is never modified."""
    yaml_path = _write_yaml(tmp_path, with_output_dir=True)
    before = yaml_path.read_text(encoding="utf-8")
    run_sweep_cli.main([str(yaml_path), "--output-dir", "sweep_output/w1_gate"])
    assert yaml_path.read_text(encoding="utf-8") == before
