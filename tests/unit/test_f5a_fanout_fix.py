# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""F5a-fanout-fix -- pure-solver device-cal placement cap (resolved count).

The F5a lift (F5A-LIFT-EFFECTIVE-PIECE3) removed the device-cal single-placement
clamp so >1 placement runs WHEN ASKED FOR (manual sets, solver-top-N, the
diversity schema). It must NOT turn the pure-solver, no-placement-specified
device-cal path into a full-device fan-out: ``find_all_placements(None)`` returns
every valid chain -- ~30,750 for a 10q circuit on Q50 (the count banked in the
Piece-1 unit log). On that one sub-path the cap defaults back to top-1, the exact
pre-lift meaning ("unspecified device-cal -> best single placement").

Two layers:
  (A) ``_solver_placement_cap`` truth table -- the always-runs guard on the cap
      decision (pure, no solver/qiskit needed).
  (B) One real-Q50 integration check -- feed the helper's cap through the genuine
      solver and assert exactly ONE placement of the ~30,750 resolves, so the
      fan-out cannot return at the resolved-count level either. Skips where the
      calibration or rustworkx is absent (runs in the LUMI unit suite, same as
      the test_placement_union real-Q50 cases).

The manual-single (-> 1) and manual-union (-> union count) device-cal sub-paths
are pinned in test_placement_union.py; between the two files every device-cal
sub-path's resolved count is asserted by count, not by path name -- the same
discipline as the noise_placement_independent flag truth table.
"""

from __future__ import annotations

import os

import pytest

from lumi_hpc_qc.sweep.sweep_engine import (
    _DEVICE_CAL_DEFAULT_SOLVER_PLACEMENTS,
    _solver_placement_cap,
)


# ── (A) cap-helper truth table -- the direct fan-out regression guard ────────

def test_pure_solver_device_cal_unset_defaults_to_top1():
    # The fix: device-cal, no manual, no max_placements -> top-1 (not ~30,750).
    assert _solver_placement_cap(None, None, wants_device_cal=True) == 1
    assert _DEVICE_CAL_DEFAULT_SOLVER_PLACEMENTS == 1


def test_pure_solver_noiseless_unset_is_unbounded():
    # Noiseless was never clamped; no placement, no cap -> solver self-selects.
    assert _solver_placement_cap(None, None, wants_device_cal=False) is None


def test_explicit_max_placements_honored_both_sources():
    # An explicit top-N is honored verbatim, device-cal or noiseless.
    assert _solver_placement_cap(None, 3, wants_device_cal=True) == 3
    assert _solver_placement_cap(None, 3, wants_device_cal=False) == 3


def test_manual_path_cap_is_passthrough():
    # With manual placements the solver is bypassed (manual) or driven by
    # solver_top_n (union); the cap is not consulted on those paths, so the
    # helper passes the explicit value through unchanged -- it must never inject
    # a top-1 into a manual or union run.
    assert _solver_placement_cap([["QB1"]], None, wants_device_cal=True) is None
    assert _solver_placement_cap([["QB1"]], 5, wants_device_cal=True) == 5


# ── (B) real-Q50 integration: capped pure-solver resolves to exactly 1 ───────

_Q50 = os.path.join(
    os.path.dirname(__file__), "..", "..",
    "examples", "q50_calibration_20260524_08c3c70f.json",
)
# 10q linear chain (the Floquet DTC connectivity) -- the circuit shape with
# ~30,750 valid placements on Q50.
_CHAIN10_EDGES = [(i, i + 1) for i in range(9)]


@pytest.mark.skipif(not os.path.exists(_Q50), reason="Q50 calibration not present")
def test_real_q50_pure_solver_device_cal_resolves_to_one():
    pytest.importorskip("rustworkx")
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver

    cal = IQMv2Adapter().load(_Q50)
    solver = GeneralPlacementSolver()
    solver.add_device(cal)
    dev = cal.device_id

    # The engine's decision for "device-cal, no manual, no max_placements".
    cap = _solver_placement_cap(None, None, wants_device_cal=True)
    assert cap == 1

    # Resolved count through the genuine solver: exactly ONE of the ~30,750, not
    # the full set. This is the assertion that would have caught the fan-out.
    resolved = solver.resolve_placements(
        circuit_edges=_CHAIN10_EDGES, circuit_qubits=10, device_id=dev,
        strategy="max_fidelity", max_placements=cap,
        manual_qubit_name_lists=None, solver_top_n=None,
    )
    assert len(resolved) == 1
    assert resolved[0].source == "solver"

    # Contrast: the unbounded path (what the lift left behind) returns the full
    # fan-out -- far more than one (the banked figure is 30,750 for this cal).
    unbounded = solver.find_all_placements(
        circuit_edges=_CHAIN10_EDGES, circuit_qubits=10,
        device_ids=[dev], strategy="max_fidelity",
    )
    assert len(unbounded) > 1
    # The capped pick is the top of the unbounded ranking (deterministic sort key).
    assert resolved[0].physical_indices == unbounded[0].physical_indices
