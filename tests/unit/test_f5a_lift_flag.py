# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""F5a lift (Piece 3) — noise_placement_independent flag semantics.

After the F5a single-placement guardrail is lifted (decision F5A-LIFT-APPROVED,
on the per-placement composition evidence F5A-VALIDATION-PROVENANCE) the
device_calibrated single-placement
clamp is removed and the flag tracks the RESOLVED placement count rather than the
(now-gone) guardrail boolean:

  - device_calibrated, 1 placement  -> True  (byte-identical to pre-lift)
  - device_calibrated, >1 placement -> False (per-placement-composed, dependent)
  - noiseless, any placement count   -> False (no per-placement noise)

This pins the truth table so the single-placement byte-identity (flag stays True)
and the multi-placement honesty (flag False) cannot silently regress.
"""

from __future__ import annotations

from lumi_hpc_qc.sweep.sweep_engine import _noise_placement_independent


def test_device_cal_single_placement_is_independent_true():
    # The pre-lift single-placement behaviour: flag True. Byte-identity hinges
    # on this staying True for exactly one placement.
    assert _noise_placement_independent("device_calibrated", 1) is True


def test_device_cal_multi_placement_is_dependent_false():
    # Lifted case: >1 placement, each composed from its own qubits -> dependent.
    assert _noise_placement_independent("device_calibrated", 2) is False
    assert _noise_placement_independent("device_calibrated", 5) is False
    assert _noise_placement_independent("device_calibrated", 30) is False


def test_noiseless_is_always_false():
    # Noiseless has no per-placement noise; the flag describes noise placement
    # dependence, so noiseless is False regardless of placement count.
    assert _noise_placement_independent("noiseless", 1) is False
    assert _noise_placement_independent("noiseless", 4) is False


def test_boundary_single_vs_multi():
    # The only True is device_calibrated at exactly 1; 1->True flips to 2->False.
    assert _noise_placement_independent("device_calibrated", 1) is True
    assert _noise_placement_independent("device_calibrated", 2) is False
