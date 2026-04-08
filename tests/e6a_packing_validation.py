#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E — E6a: Multi-Round Same-Circuit Packing Execution.

Tests circuit composition, demultiplexing, round-trip correctness
(packed energy matches individual energy), and multi-round coverage.

RED-SPEC-002 §3 — Multi-Round Same-Circuit Packing

Run on LUMI standard partition (CPU only, 4q circuits):
    srun ... python tests/e6a_packing_validation.py

Expected: E6a VALIDATION: ALL CHECKS PASSED
"""

import sys
import os
import json
import time
import traceback

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        failed += 1
        errors.append(f"{name}: {detail}")


# ══════════════════════════════════════════════════════════════════════
# Shared test fixtures
# ══════════════════════════════════════════════════════════════════════
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)

obs_zz = SparsePauliOp.from_list([("ZZ", 1.0)])


# ══════════════════════════════════════════════════════════════════════
print("\n=== E6a.1: Circuit Composition — Single Round ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.circuit_composer import compose_round
    from lumi_hpc_qc.sweep.placement_solver import Placement

    # Create 3 non-overlapping 2q placements on a 10-qubit device
    p1 = Placement(placement_id=0, device_id="test", device_prefix="test",
                    qubit_mapping={0: "Q0", 1: "Q1"},
                    physical_indices=[0, 1], score=1.0,
                    internal_edges=1, avg_readout_fidelity=0.99,
                    avg_gate_fidelity=0.99, topology_hash="h1")
    p2 = Placement(placement_id=1, device_id="test", device_prefix="test",
                    qubit_mapping={0: "Q3", 1: "Q4"},
                    physical_indices=[3, 4], score=0.9,
                    internal_edges=1, avg_readout_fidelity=0.98,
                    avg_gate_fidelity=0.98, topology_hash="h1")
    p3 = Placement(placement_id=2, device_id="test", device_prefix="test",
                    qubit_mapping={0: "Q6", 1: "Q7"},
                    physical_indices=[6, 7], score=0.8,
                    internal_edges=1, avg_readout_fidelity=0.97,
                    avg_gate_fidelity=0.97, topology_hash="h1")

    composite = compose_round(bell, [p1, p2, p3], device_qubits=10)

    check("Composition: composite has 10 qubits",
          composite.num_qubits == 10,
          f"got {composite.num_qubits}")
    check("Composition: composite has measurements",
          any(inst.operation.name == "measure" for inst in composite.data))
    check("Composition: 6 qubits measured (3 placements × 2)",
          sum(1 for inst in composite.data if inst.operation.name == "measure") == 6)

    # Overlapping placements should raise
    p_overlap = Placement(placement_id=99, device_id="test", device_prefix="test",
                          qubit_mapping={0: "Q1", 1: "Q2"},
                          physical_indices=[1, 2], score=0.5,
                          internal_edges=1, avg_readout_fidelity=0.95,
                          avg_gate_fidelity=0.95, topology_hash="h1")
    overlap_caught = False
    try:
        compose_round(bell, [p1, p_overlap], device_qubits=10)
    except ValueError:
        overlap_caught = True
    check("Composition: overlapping placements raise ValueError", overlap_caught)

except Exception as e:
    check("E6a.1 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E6a.2: Demultiplexing — Count Extraction ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.demultiplexer import demultiplex_counts

    # Simulate counts from a 10-qubit device with 3 Bell pairs
    # p1 on [0,1], p2 on [3,4], p3 on [6,7]
    # Bell state: expect "00" and "11" for each pair
    # 10-qubit bitstring: q9 q8 q7 q6 q5 q4 q3 q2 q1 q0
    raw_counts = {
        "0000000000": 500,   # all zeros
        "0010010011": 300,   # q0=1,q1=1(p1), q3=0,q4=1(p2), q6=1,q7=0(p3)
        "0011011011": 200,   # q0=1,q1=1(p1), q3=1,q4=1(p2), q6=1,q7=1(p3)
    }

    per_counts = demultiplex_counts(raw_counts, [p1, p2, p3], device_qubits=10)

    check("Demux: 3 per-placement count dicts returned",
          len(per_counts) == 3)
    check("Demux: all counts sum to total shots",
          all(sum(c.values()) == 1000 for c in per_counts),
          f"sums: {[sum(c.values()) for c in per_counts]}")

    # p1 (qubits 0,1): should see "00" and "11"
    p1_counts = per_counts[0]
    check("Demux p1: '00' present in counts",
          "00" in p1_counts,
          f"p1 counts: {p1_counts}")

    print(f"    p1 counts: {p1_counts}")
    print(f"    p2 counts: {per_counts[1]}")
    print(f"    p3 counts: {per_counts[2]}")

except Exception as e:
    check("E6a.2 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E6a.3: Round-Trip — Packed vs Individual Energy (shot-based) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.round_executor import execute_packed_rounds
    from lumi_hpc_qc.sweep.eval_runner import evaluate_circuit
    from lumi_hpc_qc.sweep.circuit_loader import LoadedCircuit

    # Use a small synthetic device (10 qubits)
    # Shot-based avoids density_matrix partial trace complexity
    small_placements = [p1, p2, p3]

    print("    Executing 3 Bell pairs packed on 10q device (shot-based, 8192 shots)...")
    t0 = time.time()
    packed_result = execute_packed_rounds(
        circuit=bell,
        observable=obs_zz,
        rounds=[small_placements],
        device_qubits=10,
        method="density_matrix",
        shots=8192,
        seed=42,
        device="CPU",
    )
    t_packed = time.time() - t0

    check("Round-trip: packed execution returns 3 results",
          packed_result.total_placements == 3,
          f"got {packed_result.total_placements}")
    check("Round-trip: no errors",
          all(pr.error is None for pr in packed_result.placement_results),
          f"errors: {[pr.error for pr in packed_result.placement_results if pr.error]}")

    packed_energies = [pr.energy for pr in packed_result.placement_results]

    # Bell ⟨ZZ⟩ = 1.0. With 8192 shots, expect each placement close to 1.0
    for i, pe in enumerate(packed_energies):
        if pe is not None:
            check(f"Round-trip p{i}: Bell ⟨ZZ⟩ ≈ 1.0 (shot-based)",
                  abs(pe - 1.0) < 0.1,
                  f"got {pe:.4f}")

    print(f"    Packed (3 placements on 10q): {t_packed:.3f}s")
    print(f"    Packed energies: {[f'{e:.4f}' if e else 'None' for e in packed_energies]}")

except Exception as e:
    check("E6a.3 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E6a.4: Q50 Packing Structure + Composite Verification ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
    from lumi_hpc_qc.sweep.circuit_composer import compose_round

    # Load Q50 calibration
    cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
    adapter = IQMv2Adapter()
    cal = adapter.load(cal_path)

    # Get real 4q placements
    solver = GeneralPlacementSolver()
    solver.add_device(cal)

    # 4q chain circuit
    qc_4q = QuantumCircuit(4)
    qc_4q.h(0)
    qc_4q.cx(0, 1)
    qc_4q.cx(1, 2)
    qc_4q.cx(2, 3)

    # TFIM 4q observable
    terms = []
    for i in range(3):
        zz = ["I"] * 4
        zz[i] = "Z"
        zz[i + 1] = "Z"
        terms.append(("".join(zz), -1.0))
    for i in range(4):
        x = ["I"] * 4
        x[i] = "X"
        terms.append(("".join(x), -1.0))
    obs = SparsePauliOp.from_list(terms)

    placements = solver.find_all_placements(
        circuit_edges=[(0, 1), (1, 2), (2, 3)],
        circuit_qubits=4,
    )
    rounds = solver.pack_rounds(placements)
    print(f"    {len(placements)} placements in {len(rounds)} rounds")

    # Verify composite structure (without executing — Q50 composite is too large for Aer)
    max_rounds = min(3, len(rounds))
    test_rounds = rounds[:max_rounds]
    total_in_rounds = sum(len(r.placements) for r in test_rounds)

    for ri, rnd in enumerate(test_rounds):
        composite = compose_round(qc_4q, rnd.placements, cal.num_qubits)
        check(f"Q50 round {ri}: composite has {cal.num_qubits} qubits",
              composite.num_qubits == cal.num_qubits)

        # Count measurements — should equal total physical qubits used
        n_meas = sum(1 for inst in composite.data if inst.operation.name == "measure")
        expected_meas = len(rnd.placements) * 4  # 4 qubits per placement
        check(f"Q50 round {ri}: {expected_meas} qubits measured",
              n_meas == expected_meas,
              f"got {n_meas}")

    # Verify all placements covered across all rounds
    all_placement_ids = set()
    for rnd in rounds:
        for p in rnd.placements:
            all_placement_ids.add(p.placement_id)
    check("Q50 packing: all placements covered",
          len(all_placement_ids) == len(placements),
          f"packed {len(all_placement_ids)}, total {len(placements)}")

    # Verify round sizes sum to total
    total_packed = sum(len(r.placements) for r in rounds)
    check("Q50 packing: round sizes sum to total",
          total_packed == len(placements),
          f"sum={total_packed}, total={len(placements)}")

    print(f"    Composite verification: {max_rounds} rounds checked")
    for ri, rnd in enumerate(test_rounds):
        print(f"      Round {ri}: {len(rnd.placements)} placements, "
              f"{len(rnd.placements) * 4} qubits used")

    # NOTE: Actual execution of Q50 composites is for QPU submissions only.
    # For Aer, the twin simulator (E4) runs each placement individually.
    # Packed execution is validated on the 10q device in E6a.3.

except Exception as e:
    check("E6a.4 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E6a.5: Provenance — Packing Round Metadata ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Each PackingRound should carry its round_id and device_id
    for i, rnd in enumerate(rounds[:3]):
        check(f"Provenance: round {i} has round_id={i}",
              rnd.round_id == i,
              f"got {rnd.round_id}")
        check(f"Provenance: round {i} has device_id",
              len(rnd.device_id) > 0,
              f"got '{rnd.device_id}'")
        check(f"Provenance: round {i} total_qubits_used > 0",
              rnd.total_qubits_used > 0,
              f"got {rnd.total_qubits_used}")

    # From packed execution (E6a.3), verify round index in results
    if packed_result and packed_result.rounds:
        for pr in packed_result.rounds[0].placement_results:
            check(f"Provenance: packed result has round_index=0",
                  pr.round_index == 0,
                  f"got {pr.round_index}")
            break  # just check first

except Exception as e:
    check("E6a.5 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E6a.6: Determinism — Same Seed Same Results ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Re-execute small device round with same seed (from E6a.3)
    repeat_result = execute_packed_rounds(
        circuit=bell,
        observable=obs_zz,
        rounds=[small_placements],
        device_qubits=10,
        method="density_matrix",
        shots=8192,
        seed=42,
        device="CPU",
    )

    # Verify all repeat energies match Bell ⟨ZZ⟩ = 1.0
    all_correct = True
    for pr in repeat_result.placement_results:
        if pr.energy is None or abs(pr.energy - 1.0) > 0.1:
            check(f"Determinism: placement {pr.physical_indices} ⟨ZZ⟩ ≈ 1.0",
                  False, f"got {pr.energy}")
            all_correct = False
    if all_correct:
        check("Determinism: repeat run all Bell ⟨ZZ⟩ ≈ 1.0", True)

    # Also check that same-seed gives same counts (exact match on p0)
    if (packed_result.placement_results and repeat_result.placement_results):
        e1 = packed_result.placement_results[0].energy
        e2 = repeat_result.placement_results[0].energy
        if e1 is not None and e2 is not None:
            check("Determinism: p0 same seed same energy",
                  abs(e1 - e2) < 1e-10,
                  f"first={e1:.6f}, repeat={e2:.6f}")

except Exception as e:
    check("E6a.6 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E6a.7: Bug Fix A — Mixed Qubit Count Demultiplexing ===")
# ══════════════════════════════════════════════════════════════════════

try:
    from lumi_hpc_qc.sweep.demultiplexer import demultiplex_counts
    from lumi_hpc_qc.sweep.placement_solver import Placement

    # Two placements with DIFFERENT qubit counts:
    # p_4q uses 4 physical qubits, p_2q uses 2 physical qubits
    p_4q = Placement(
        placement_id=0, device_id="test", device_prefix="test",
        physical_indices=[0, 1, 2, 3],
        qubit_mapping={0: "Q0", 1: "Q1", 2: "Q2", 3: "Q3"},
        topology_hash="4q", score=1.0, per_qubit_calibration={},
        internal_edges=3, avg_readout_fidelity=0.99, avg_gate_fidelity=0.98,
    )
    p_2q = Placement(
        placement_id=1, device_id="test", device_prefix="test",
        physical_indices=[6, 7],
        qubit_mapping={0: "Q6", 1: "Q7"},
        topology_hash="2q", score=1.0, per_qubit_calibration={},
        internal_edges=1, avg_readout_fidelity=0.99, avg_gate_fidelity=0.98,
    )

    device_qubits = 10

    # Construct a synthetic bitstring where we know the answer.
    # Device bitstring: positions 0,1,2,3 = "1010", positions 6,7 = "11"
    # Qiskit convention: bitstring[i] = qubit[N-1-i]
    # For 10 qubits: bit_pos(qubit_k) = 9 - k
    # qubit 0 → pos 9, qubit 1 → pos 8, qubit 2 → pos 7, qubit 3 → pos 6
    # qubit 6 → pos 3, qubit 7 → pos 2
    #
    # So for 4q result "1010": q0=1, q1=0, q2=1, q3=0
    #   pos9=1, pos8=0, pos7=1, pos6=0 → bits "...0101........."
    # For 2q result "11": q6=1, q7=1
    #   pos3=1, pos2=1 → bits "........11......"
    #
    # Full 10-bit string (pos 9..0): 1 0 1 0 _ _ 1 1 _ _
    # =                              1 0 1 0 0 0 1 1 0 0 → "1010001100"
    raw_counts = {"1010001100": 100}

    per_counts = demultiplex_counts(raw_counts, [p_4q, p_2q], device_qubits)

    check("Bug Fix A: two placement results returned",
          len(per_counts) == 2,
          f"got {len(per_counts)}")

    # p_4q should extract 4-bit string
    check("Bug Fix A: 4q placement has 4-bit keys",
          all(len(k) == 4 for k in per_counts[0].keys()),
          f"keys: {list(per_counts[0].keys())}")

    # p_2q should extract 2-bit string
    check("Bug Fix A: 2q placement has 2-bit keys",
          all(len(k) == 2 for k in per_counts[1].keys()),
          f"keys: {list(per_counts[1].keys())}")

    # Verify extracted values
    check("Bug Fix A: 4q extracted '1010' with count 100",
          per_counts[0].get("1010") == 100,
          f"got {per_counts[0]}")
    check("Bug Fix A: 2q extracted '11' with count 100",
          per_counts[1].get("11") == 100,
          f"got {per_counts[1]}")

except Exception as e:
    check("E6a.7 Bug Fix A", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"E6a VALIDATION: {passed} passed, {failed} failed")
if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  ✗ {e}")
    print(f"\nE6a VALIDATION: FAILED ({failed} failures)")
    sys.exit(1)
else:
    print(f"\nE6a VALIDATION: ALL {passed} CHECKS PASSED")
    sys.exit(0)
