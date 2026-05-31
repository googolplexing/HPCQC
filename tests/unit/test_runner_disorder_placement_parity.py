# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Patch-18 parity guards (RED-REVIEW-PATCH18 §2).

Patch 18 gave ``floquet_runner_v2`` a ``--disorder-file`` and a
``--physical-qubits`` so the regenerated reference arm and the BYO sweep arm
consume the SAME banked disorder on the SAME pinned placement through the SAME
``prepare_simulation`` seam. Disorder + placement are made identical *by reuse*
(``resolve_disorder`` / ``resolve_placements``), so those hold by construction.

But the two arms still BUILD their circuits through different functions —
``floquet_runner_v2.build_circuit`` vs the BYO factory
``examples/byo/floquet_dtc.py:build_circuit`` (driven by the worker via
``load_circuit`` + ``assemble_build_kwargs``). Identical disorder is necessary
but NOT sufficient for identical circuits: the two builders must also APPLY that
disorder identically. Today they do (Red traced them gate-for-gate), and the
pure-z_comb gate now RESTS on that equivalence — but nothing guarded it. A
future edit to either builder (a reordered gate, a changed tail convention, an
``h_x`` tweak) would silently break parity and surface as an "unexplained
residual," which under RED ruling #3 is an engine-finding/blocker. This file is
the drift canary that keeps "builder" on the structurally-identical side of
that line — the same way ``test_canonical_placement_guard`` closed the F5
emission-order gap.

Three groups (RED-REVIEW-PATCH18 §2):
  1. builder equivalence: runner ``build_circuit`` == BYO factory, same file
     disorder, gate-for-gate (the new load-bearing fact);
  2. ``--disorder-file`` round-trip: the runner consumes the file's exact
     hz/Jzz/init_bit_array and those values land in the circuit's gates;
  3. ``--physical-qubits`` parse + pinned placement: the canonical name list
     resolves through the solver seam to the pinned logical->name mapping, and a
     mis-ordered list fails loud (no silent wrong-qubit run).

Needs real qiskit (circuit construction) + the calibration; like the byo-wiring
and canonical-placement guards it runs on LUMI with the unit suite, NOT under
the offline (stubbed) harness. No Aer simulation and no rustworkx placement
search are exercised, so it is cheap.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

# floquet_runner_v2 lives at the repo ROOT (it is not part of the lumi_hpc_qc
# package), so put the root on sys.path before importing it. Anchored to this
# file, not the CWD, so collection order / invocation dir cannot break it.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import floquet_runner_v2 as runner  # noqa: E402

from lumi_hpc_qc.plugins.registry import PluginRegistry  # noqa: E402
from lumi_hpc_qc.sweep.byo_sweep import (  # noqa: E402
    assemble_build_kwargs,
    resolve_disorder,
)
from lumi_hpc_qc.sweep.circuit_loader import (  # noqa: E402
    extract_disorder_signature,
    load_circuit,
)
from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver  # noqa: E402

_DISORDER = os.path.join(_ROOT, "examples/byo/floquet_disorder_q10.json")
_CAL = os.path.join(_ROOT, "examples/q50_calibration_20260524_08c3c70f.json")
_FACTORY = os.path.join(_ROOT, "examples/byo/floquet_dtc.py")

# The pinned-path Option-1 invariants the W1.6 gate runs under (file _meta +
# runner defaults): q10, initial_state 3, epsilon 0.03, canonical top_1 order.
_NUM_QUBITS = 10
_INITIAL_STATE = 3
_EPSILON = 0.03
_CHAIN = [(i, i + 1) for i in range(_NUM_QUBITS - 1)]  # q10 open linear chain
_CANONICAL = [
    "QB11", "QB5", "QB6", "QB7", "QB13", "QB21", "QB29", "QB28", "QB27", "QB26",
]


# ── helpers ────────────────────────────────────────────────────────────────
def _load_instance(instance_id: int) -> dict:
    """The file disorder for one instance, via the SAME call the runner makes."""
    resolved, _ = resolve_disorder(
        {"source": "file", "file": _DISORDER},
        [instance_id],
        num_qubits=_NUM_QUBITS,
        configured_initial_state=_INITIAL_STATE,
    )
    return resolved[instance_id]


def _runner_circuit(inst: dict, num_kicks: int):
    """Build via the runner exactly as run_one_instance does in file mode
    (np.asarray on the file arrays, h_x = (1-epsilon)*pi)."""
    hz = np.asarray(inst["hz_angles"], dtype=float)
    jzz = np.asarray(inst["Jzz_angles"], dtype=float)
    init = list(inst["init_bit_array"])
    h_x = (1 - _EPSILON) * np.pi
    return runner.build_circuit(num_kicks, hz, jzz, init, _NUM_QUBITS, h_x)


def _byo_circuit(inst: dict, num_kicks: int):
    """Build via the BYO factory through the SAME seam the worker uses
    (assemble_build_kwargs over fixed/disorder/grid, then load_circuit)."""
    build_kwargs = assemble_build_kwargs(
        {"epsilon": _EPSILON, "num_qubits": _NUM_QUBITS},  # fixed
        inst,                                              # disorder instance
        {"num_kicks": num_kicks},                          # grid point
    )
    loaded = load_circuit(
        script_file=_FACTORY,
        script_function="build_circuit",
        script_params=build_kwargs,
    )
    return loaded.circuit


def _ops(qc) -> list:
    """Ordered (name, qubit-indices, clbit-indices, params) for every gate —
    a full gate-for-gate signature. Params rounded to absorb any float64-vs-
    float repr difference between the two argument paths (the values are the
    same JSON-parsed doubles through identical arithmetic, so this only guards
    against representation, never masks a real divergence)."""
    out = []
    for ci in qc.data:
        qubits = tuple(qc.find_bit(q).index for q in ci.qubits)
        clbits = tuple(qc.find_bit(c).index for c in ci.clbits)
        params = tuple(round(float(p), 12) for p in ci.operation.params)
        out.append((ci.operation.name, qubits, clbits, params))
    return out


def _solver_for_cal():
    cal_json = json.load(open(_CAL))
    reg = PluginRegistry()
    reg.discover()  # engine does this at construction (sweep_engine.py:1313)
    adapter = reg.get_calibration_adapter(cal_json.get("adapter", "iqm_v2"))
    device_cal = adapter.load(_CAL)
    solver = GeneralPlacementSolver()
    solver.add_device(device_cal)
    return solver, device_cal


# ── 1. builder equivalence (the new load-bearing fact) ──────────────────────
@pytest.mark.parametrize("num_kicks", [1, 2, 3])
def test_runner_and_byo_factory_build_identical_circuits(num_kicks):
    """Gate-for-gate identity between the runner builder and the BYO factory on
    the same file disorder. This is what the pure-z_comb gate rests on; if it
    ever fails, parity is broken at the builder and the gate residual would
    masquerade as an engine finding (RED ruling #3)."""
    inst = _load_instance(0)
    runner_ops = _ops(_runner_circuit(inst, num_kicks))
    byo_ops = _ops(_byo_circuit(inst, num_kicks))
    assert runner_ops == byo_ops, (
        f"runner build_circuit and BYO factory diverged at num_kicks={num_kicks}; "
        "the W1.6 gate's near-bit-level expectation rests on their equivalence "
        "(RED-REVIEW-PATCH18 §2 drift canary)."
    )


# ── 2. --disorder-file round-trip ───────────────────────────────────────────
def test_disorder_file_roundtrip_values():
    """The runner's load path returns the file's exact per-instance arrays."""
    raw = json.load(open(_DISORDER))["instances"]["0"]
    inst = _load_instance(0)
    assert list(inst["hz_angles"]) == list(raw["hz_angles"])
    assert list(inst["Jzz_angles"]) == list(raw["Jzz_angles"])
    assert list(inst["init_bit_array"]) == list(raw["init_bit_array"])


def test_disorder_file_values_land_in_runner_circuit():
    """The file's hz/Jzz actually reach the circuit gates: rz on wire w carries
    hz[w], rzz on bond (w,w+1) carries Jzz[w]; the tail Jzz[n-1] is unused (only
    n-1 bonds); all-zero init_bit_array means no X-init gates."""
    inst = _load_instance(0)
    hz = [float(x) for x in inst["hz_angles"]]
    jzz = [float(x) for x in inst["Jzz_angles"]]
    sig = extract_disorder_signature(_runner_circuit(inst, num_kicks=1))
    for w in range(_NUM_QUBITS):
        assert sig["rz"][(w,)][0] == pytest.approx(hz[w])
    for w in range(_NUM_QUBITS - 1):
        assert sig["rzz"][(w, w + 1)][0] == pytest.approx(jzz[w])
    assert list(inst["init_bit_array"]) == [0] * _NUM_QUBITS
    assert "x" not in {ci.operation.name for ci in _runner_circuit(inst, 1).data}


# ── 3. --physical-qubits parse + pinned placement ───────────────────────────
def test_physical_qubits_cli_parse_contract():
    """The runner's CLI normalizer, exercised directly (patch 18c lifted it into
    `_parse_physical_qubits_cli`). Comma-split, strip, drop blanks; None/"" ->
    None (free-layout default); an all-blank value -> [] (NOT None) so a
    malformed flag fails loud downstream rather than silently free-layouting."""
    parse = runner._parse_physical_qubits_cli
    assert parse(",".join(_CANONICAL)) == _CANONICAL
    assert parse(" QB11 , QB5 ,") == ["QB11", "QB5"]
    assert parse(None) is None
    assert parse("") is None
    assert parse(" , ") == []


def test_canonical_physical_qubits_resolve_to_pinned_mapping():
    """The canonical name list resolves through the SAME seam the runner calls
    (resolve_placements, solver bypassed) to logical i -> the i-th name — i.e.
    the runner's prep_phys_qubits == _CANONICAL."""
    solver, dev = _solver_for_cal()
    pls = solver.resolve_placements(
        circuit_edges=_CHAIN,
        circuit_qubits=_NUM_QUBITS,
        device_id=dev.device_id,
        strategy="max_fidelity",
        manual_qubit_name_lists=[_CANONICAL],
    )
    assert len(pls) == 1
    mapping = [pls[0].qubit_mapping[i] for i in range(_NUM_QUBITS)]
    assert mapping == _CANONICAL


def test_misordered_physical_qubits_fail_loud():
    """A mis-ordered list whose chain breaks device adjacency must raise at
    resolution — so a typo'd / wrongly-ordered --physical-qubits cannot silently
    run on the wrong qubits (RED-REVIEW-PATCH18 §5). Swapping QB11<->QB13
    (indices 0,4) breaks edges (QB13,QB5)/(QB7,QB11)/(QB11,QB21) on cal
    08c3c70f."""
    bad = list(_CANONICAL)
    bad[0], bad[4] = bad[4], bad[0]
    solver, dev = _solver_for_cal()
    with pytest.raises(ValueError):
        solver.resolve_placements(
            circuit_edges=_CHAIN,
            circuit_qubits=_NUM_QUBITS,
            device_id=dev.device_id,
            strategy="max_fidelity",
            manual_qubit_name_lists=[bad],
        )
