"""Permanent guard: device-calibrated noise is composed PER PLACEMENT.

This is the CI-level half of RED-DIRECTIVE Piece 2 — the composition invariant
(assert (a): each placement's model is built from its own qubits, no
cross-contamination). It needs qiskit_aer, so it runs in the LUMI container with
the rest of the unit suite. The cal-tracking half (assert (b)) needs a real sweep
and lives in the harness scripts/validate_per_placement_f5a.py.

The check logic is imported from the harness (single source of truth) rather than
re-implemented here, so the CI guard and the evidence generator cannot drift.
"""
from __future__ import annotations

import importlib.util
import os

import pytest

# Path-load the harness (scripts/ is not an importable package).
_HARNESS = os.path.join(
    os.path.dirname(__file__), "..", "..", "scripts", "validate_per_placement_f5a.py")
_spec = importlib.util.spec_from_file_location("f5a_harness", _HARNESS)
f5a = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(f5a)

_CAL = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "examples", "q50_calibration_20260524_08c3c70f.json")


def test_per_placement_composition_no_cross_contamination():
    """Each chain's device-cal model is built from its own qubits (assert (a))."""
    result = f5a.check_composition(_CAL)  # raises AssertionError on any violation
    assert result["passed"]
    assert result["disjoint"], "validation chains must be disjoint qubit sets"
    assert result["t2_contrast_ratio"] > 2.0, "need a real T2 contrast to be decisive"
    # The two per-qubit T2 vectors must be elementwise distinct (already asserted
    # inside check_composition; re-stated here as the headline CI guarantee).
    hi = result["high_per_qubit_t2_us"]
    lo = result["low_per_qubit_t2_us"]
    assert all(abs(a - b) > 1e-9 for a, b in zip(hi, lo))


def test_observable_independence_statement_present():
    """The lift-scope statement must exist so it can't be silently dropped."""
    s = f5a.OBSERVABLE_INDEPENDENCE_STATEMENT.lower()
    assert "no observable" in s and "echo" in s and "autocorr" in s
    assert "observable-general" in s
