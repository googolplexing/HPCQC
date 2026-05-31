"""D7 increment 2 — multi-observable BYO foundation.

Guards the two offline-testable invariants Red required for the staging:
  1. the legacy-vs-observable PATH decision (one helper, both consumers);
  2. the BACKWARD-COMPAT invariant — absent ``observables`` synthesizes the
     single "default" family and the expansion is structurally identical to the
     pre-D7 single-observable path (same circuit_function, "default" name, ""
     subpath), so the W1.6 gate path is untouched.

The full HDF5/.dat byte-identity is an on-LUMI end-to-end check; here we pin the
structural invariants the engine wiring rests on.
"""
from __future__ import annotations

import pytest

from lumi_hpc_qc.sweep.byo_observable import (
    DEFAULT_OBSERVABLE_NAME,
    byo_observable_subpath,
)
from lumi_hpc_qc.sweep.sweep_engine import (
    _parse_observables,
    expand_grid,
    parse_sweep_config,
)


# ── 1. the legacy-vs-observable path helper ──────────────────────────────────

def test_subpath_default_is_legacy_empty():
    # The synthesized single observable must add NO path level (byte-identical).
    assert byo_observable_subpath(DEFAULT_OBSERVABLE_NAME) == ""


def test_subpath_named_adds_one_level():
    assert byo_observable_subpath("echo") == "/echo"
    assert byo_observable_subpath("autocorr") == "/autocorr"


# ── 2. _parse_observables synthesis + validation ─────────────────────────────

def test_parse_observables_absent_synthesizes_default():
    assert _parse_observables(None, "build_circuit") == (
        (DEFAULT_OBSERVABLE_NAME, "build_circuit"),
    )


def test_parse_observables_absent_uses_given_circuit_function():
    assert _parse_observables(None, "build_circuit_echo") == (
        (DEFAULT_OBSERVABLE_NAME, "build_circuit_echo"),
    )


def test_parse_observables_explicit_pair():
    raw = [
        {"name": "autocorr", "function": "build_circuit"},
        {"name": "echo", "function": "build_circuit_echo"},
    ]
    assert _parse_observables(raw, "build_circuit") == (
        ("autocorr", "build_circuit"),
        ("echo", "build_circuit_echo"),
    )


@pytest.mark.parametrize("bad", [
    [],                                              # empty list
    [{"name": "x"}],                                 # missing function
    [{"function": "f"}],                             # missing name
    [{"name": "", "function": "f"}],                 # empty name
    [{"name": "x", "function": ""}],                 # empty function
    "build_circuit",                                 # not a list
])
def test_parse_observables_rejects_malformed(bad):
    with pytest.raises(ValueError):
        _parse_observables(bad, "build_circuit")


def test_parse_observables_rejects_duplicate_name():
    raw = [
        {"name": "a", "function": "build_circuit"},
        {"name": "a", "function": "build_circuit_echo"},
    ]
    with pytest.raises(ValueError):
        _parse_observables(raw, "build_circuit")


def test_parse_observables_rejects_reserved_default_name():
    # "default" is reserved for the synthesized single-observable case.
    raw = [{"name": DEFAULT_OBSERVABLE_NAME, "function": "build_circuit"}]
    with pytest.raises(ValueError):
        _parse_observables(raw, "build_circuit")


# ── 3. parse_sweep_config: synthesis vs declared ─────────────────────────────

def _byo_exp(**overrides):
    exp = {
        "type": "byo_circuit",
        "circuit_script": "examples/byo/floquet_dtc_echo.py",
        "fixed": {"num_qubits": 10, "epsilon": 0.03},
        "grid": {"num_kicks": {"range": [0, 4]}},
        "disorder": {"source": "file",
                     "file": "examples/byo/floquet_disorder_q10_echo_ak10.json",
                     "initial_state": 3},
        "disorder_gates": ["rz", "rzz"],
        "seed_list": [0, 1],
        "shots": 100,
        "noise_configs": ["noiseless"],
    }
    exp.update(overrides)
    return {"sweep": {"experiments": [exp],
                      "calibrations": ["examples/q50_calibration_20260524_08c3c70f.json"]}}


def test_parse_synthesizes_default_when_observables_absent():
    cfg = parse_sweep_config(_byo_exp())
    assert cfg.experiments[0].observables == (
        (DEFAULT_OBSERVABLE_NAME, "build_circuit"),
    )


def test_parse_reads_declared_observables():
    cfg = parse_sweep_config(_byo_exp(observables=[
        {"name": "autocorr", "function": "build_circuit"},
        {"name": "echo", "function": "build_circuit_echo"},
    ]))
    assert cfg.experiments[0].observables == (
        ("autocorr", "build_circuit"),
        ("echo", "build_circuit_echo"),
    )


# ── 4. expansion: backward-compat default + per-observable split (LUMI) ───────
# These call expand_grid -> load_factory (qiskit), so they run with the unit
# suite on LUMI, not under the offline harness.

def test_expand_single_default_all_tasks_legacy():
    cfg = parse_sweep_config(_byo_exp())
    tasks = expand_grid(cfg)
    assert tasks, "no tasks expanded"
    # Every task is the synthesized default family, unchanged circuit_function.
    assert all(t.observable_name == DEFAULT_OBSERVABLE_NAME for t in tasks)
    assert all(t.circuit_function == "build_circuit" for t in tasks)
    # ... and the default subpath is empty (legacy layout).
    assert byo_observable_subpath(tasks[0].observable_name) == ""


def test_expand_two_observables_splits_tasks_per_family():
    cfg = parse_sweep_config(_byo_exp(observables=[
        {"name": "autocorr", "function": "build_circuit"},
        {"name": "echo", "function": "build_circuit_echo"},
    ]))
    tasks = expand_grid(cfg)
    by_obs = {}
    for t in tasks:
        by_obs.setdefault(t.observable_name, set()).add(t.circuit_function)
    # Two families, each pinned to its own factory function.
    assert by_obs == {
        "autocorr": {"build_circuit"},
        "echo": {"build_circuit_echo"},
    }
    # Same task count per family (same seeds × grid × cal × envs).
    n_auto = sum(1 for t in tasks if t.observable_name == "autocorr")
    n_echo = sum(1 for t in tasks if t.observable_name == "echo")
    assert n_auto == n_echo and n_auto > 0
