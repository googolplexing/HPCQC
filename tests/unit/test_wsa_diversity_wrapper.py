# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Discoverability wrapper for the Workstream A diversity-equivalence harness.

The real test lives in tests/f5a_diversity_equivalence_validation.py and runs
the solver's disjoint selection plus K+1 tiny aer sweeps. This wrapper makes the
guard VISIBLE in the unit run (RED-RULING-WORKSTREAM-A §6): `pytest tests/unit`
lists it as SKIPPED with an actionable reason, without paying the aer cost on
every run.

Gated on HPCQC_RUN_SLOW=1: the gate is on EXECUTION, not visibility. The actual
gate is the dedicated slurm job (tests/slurm_wsa_diversity.sh), which calls the
harness main() DIRECTLY and is never gated — so the gate that matters can't be a
no-op skip. Run on demand with:
    HPCQC_RUN_SLOW=1 pytest tests/unit/test_wsa_diversity_wrapper.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("HPCQC_RUN_SLOW"),
    reason="set HPCQC_RUN_SLOW=1 to run the WSA diversity-equivalence harness (needs aer)",
)


def test_wsa_diversity_equivalence_harness():
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("h5py")
    harness_path = (
        Path(__file__).resolve().parents[1]
        / "f5a_diversity_equivalence_validation.py"
    )
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "f5a_diversity_equivalence_validation", harness_path
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.main() == 0, (
        "WSA diversity-equivalence harness reported failures (see captured stdout)"
    )
