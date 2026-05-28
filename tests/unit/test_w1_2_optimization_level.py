# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""W1.2 — configurable optimization_level YAML field with provenance (CFG-2).

Verifies that:
  (1) the new `optimization_level` field on SweepExperimentConfig and SweepTask
      defaults to None (which resolves to 3 at the execution site, matching
      the historical hardcode and the banked-reference / gate-2 pin);
  (2) the parse-time validator (_parse_optimization_level) accepts 0..3,
      rejects out-of-range and non-integer values with a clear message;
  (3) the resolution expression at the execution site treats None as 3 and
      passes any explicit 0..3 value through unchanged.

Provenance threading (the per-result HDF5 attr "optimization_level") and the
non-default health note are observed by the LUMI canary at W1.3 and gate-2
verification at W1.6; this file's job is to lock the schema + the validator
down so those downstream checks have a stable contract to assert against.

Per RED-RESP-W1-PARALLELISM-AND-OOM-ROOTCAUSE-v1.4 Q3 ACCEPT: gate-2 pins 3
(the banked-reference lineage); changing the level changes the transpiled
gate set + scheduling and therefore the applied noise model, so it is
physics-affecting under noise and is NOT a free performance knob.
"""

from __future__ import annotations

from lumi_hpc_qc.sweep.sweep_engine import (
    SweepExperimentConfig,
    SweepTask,
    _parse_optimization_level,
)


# ── Defaults: None on both config and task ─────────────────────────────────

def test_experiment_config_default_optimization_level_is_none():
    """Absent YAML field -> None -> resolved to 3 at the execution site."""
    exp = SweepExperimentConfig()
    assert exp.optimization_level is None


def test_sweep_task_default_optimization_level_is_none():
    """SweepTask carries None by default; resolution to 3 happens at exec."""
    task = SweepTask()
    assert task.optimization_level is None


# ── Validator: accepts 0..3, raw int and stringified ──────────────────────

def test_parse_optimization_level_returns_none_for_none():
    """YAML omits the field -> parser returns None (default sentinel)."""
    assert _parse_optimization_level(None) is None


def test_parse_optimization_level_accepts_each_valid_int():
    for level in (0, 1, 2, 3):
        assert _parse_optimization_level(level) == level, (
            f"valid level {level} must be accepted"
        )


def test_parse_optimization_level_accepts_stringified_int():
    """YAML occasionally hands us strings (esp. when the field comes from
    a templated config); int(raw) should coerce cleanly."""
    assert _parse_optimization_level("2") == 2
    assert _parse_optimization_level("0") == 0


# ── Validator: rejects out-of-range and non-integer values ────────────────

def test_parse_optimization_level_rejects_above_3():
    """opt_level=4 is invalid in Qiskit; we must fail at parse time, not at
    run time deep inside the transpile() call."""
    try:
        _parse_optimization_level(4)
    except ValueError as e:
        assert "0, 1, 2, 3" in str(e), (
            f"error must enumerate valid range; got: {e}"
        )
        return
    raise AssertionError("ValueError expected for opt_level=4")


def test_parse_optimization_level_rejects_negative():
    try:
        _parse_optimization_level(-1)
    except ValueError as e:
        assert "0, 1, 2, 3" in str(e)
        return
    raise AssertionError("ValueError expected for opt_level=-1")


def test_parse_optimization_level_rejects_non_integer_string():
    """A typo in the YAML (e.g. 'three') must raise — not silently become 3."""
    try:
        _parse_optimization_level("three")
    except ValueError as e:
        assert "integer" in str(e).lower(), f"error must mention integer; got: {e}"
        return
    raise AssertionError("ValueError expected for non-integer string")


def test_parse_optimization_level_rejects_float():
    """A float like 2.5 is not a valid Qiskit optimization level — must fail.

    int(2.5) == 2 silently in Python, which would let a typo'd YAML
    `optimization_level: 2.5` pass parsing as 2 — a soft-accept hazard. Per
    Q3 ACCEPT's "honored exactly as specified" wording, the validator must
    reject floats explicitly rather than truncate them.
    """
    try:
        _parse_optimization_level(2.5)
    except ValueError as e:
        assert "float" in str(e).lower() or "integer" in str(e).lower(), (
            f"error must mention the type or integer requirement; got: {e}"
        )
        return
    raise AssertionError("ValueError expected for float 2.5 (no silent truncation)")


def test_parse_optimization_level_rejects_bool():
    """bool is a subclass of int in Python (True == 1, False == 0); accepting
    booleans would let a typo'd `optimization_level: true` parse as level 1.
    Must reject explicitly."""
    for bad in (True, False):
        try:
            _parse_optimization_level(bad)
        except ValueError as e:
            assert "bool" in str(e).lower() or "integer" in str(e).lower()
            continue
        raise AssertionError(f"ValueError expected for bool {bad!r}")


# ── Resolution: None -> 3, explicit value passed through ──────────────────
#
# Mirrors the execution-site expression at sweep_engine.py
#   exp_opt_level = (
#       representative.optimization_level
#       if representative.optimization_level is not None else 3
#   )
# We pin the resolution contract here so the W1.3 worker code (which will
# read task.optimization_level inside the forkserver child) can rely on it.

def _resolve(task_opt_level: int | None) -> int:
    """Re-implements the execution-site resolution; CFG-2 default = 3."""
    return task_opt_level if task_opt_level is not None else 3


def test_resolve_none_is_default_3():
    """The default behaviour preserves the historical hardcode."""
    assert _resolve(None) == 3


def test_resolve_passes_explicit_values_through():
    """Explicit 0..3 must be honored exactly (gate-2 pins 3 by omission;
    a smoke-test sweep at opt_level=0 must actually use 0)."""
    for level in (0, 1, 2, 3):
        assert _resolve(level) == level


# ── Task carries the field through the dataclass round-trip ───────────────

def test_sweep_task_carries_explicit_optimization_level():
    """SweepTask(optimization_level=N) round-trips and is readable downstream."""
    task = SweepTask(optimization_level=2)
    assert task.optimization_level == 2
    # And the resolution treats it correctly:
    assert _resolve(task.optimization_level) == 2


def test_experiment_config_carries_explicit_optimization_level():
    """SweepExperimentConfig(optimization_level=N) survives construction."""
    exp = SweepExperimentConfig(optimization_level=1)
    assert exp.optimization_level == 1
