# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Discoverability wrapper for the F5a no-cross-talk integration harness.

The real test lives in tests/f5a_no_crosstalk_validation.py and runs three tiny
aer sweeps (~2 min). This wrapper exists so the guard is VISIBLE in the unit run
(RED-RULING-F5A-NO-CROSSTALK-CI Q3): anyone running `pytest tests/unit` sees it
listed as SKIPPED with an actionable reason — "standing guard, not a script
someone has to remember" — without paying the ~2-min aer cost on every run.

Gated on HPCQC_RUN_SLOW=1 (RED-RULING Q1): the gate is on EXECUTION, not
visibility — a skipped-with-clear-reason test is still discoverable in the unit
run. This keeps `pytest tests/unit` in its cheap-and-reflexive regime so people
don't route around the suite. The actual CI gate before the VIP banks is the
dedicated slurm job (tests/slurm_f5a_crosstalk.sh), which calls the harness
main() DIRECTLY and is never gated — so the gate that matters can't be a no-op
skip. Run on demand with: HPCQC_RUN_SLOW=1 pytest tests/unit/test_f5a_no_crosstalk_wrapper.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("HPCQC_RUN_SLOW"),
    reason="set HPCQC_RUN_SLOW=1 to run the F5a no-cross-talk harness (~2 min, needs aer)",
)


def test_f5a_no_crosstalk_harness():
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("h5py")
    # Import the standalone harness by path (tests/ is not a package).
    harness_path = Path(__file__).resolve().parents[1] / "f5a_no_crosstalk_validation.py"
    import importlib.util
    spec = importlib.util.spec_from_file_location("f5a_no_crosstalk_validation", harness_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    assert mod.main() == 0, "F5a no-cross-talk harness reported failures (see captured stdout)"
