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
print("\n=== E6a.3: Round-Trip — Packed vs Individual Energy (small device) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.round_executor import execute_packed_rounds
    from lumi_hpc_qc.sweep.eval_runner import evaluate_circuit
    from lumi_hpc_qc.sweep.circuit_loader import LoadedCircuit

    # Use a small synthetic device (10 qubits) so exact DM is feasible
    # 3 non-overlapping 2q placements
    small_placements = [p1, p2, p3]  # from E6a.1 on 10-qubit device

    print("    Executing 3 Bell pairs packed on 10q device (exact DM)...")
    t0 = time.time()
    packed_result = execute_packed_rounds(
        circuit=bell,
        observable=obs_zz,
        rounds=[small_placements],
        device_qubits=10,
        method="density_matrix",
        shots=0,
        seed=42,
        device="CPU",
    )
    t_packed = time.time() - t0

    check("Round-trip: packed execution returns 3 results",
          packed_result.total_placements == 3,
          f"got {packed_result.total_placements}")

    # Execute each individually for comparison
    individual_energies = []
    for p in small_placements:
        loaded = LoadedCircuit(
            circuit=bell, num_qubits=2, num_parameters=0,
            is_parameterized=False, connectivity=[(0, 1)],
            source="test",
        )
        result = evaluate_circuit(
            loaded, observable=obs_zz,
            method="density_matrix", shots=0, seed=42, device="CPU",
        )
        individual_energies.append(result.energy)

    packed_energies = [pr.energy for pr in packed_result.placement_results]

    for i, (pe, ie) in enumerate(zip(packed_energies, individual_energies)):
        if pe is not None and ie is not None:
            diff = abs(pe - ie)
            check(f"Round-trip p{i}: |packed - individual| < 1e-6",
                  diff < 1e-6,
                  f"packed={pe:.6f}, individual={ie:.6f}, diff={diff:.2e}")

    print(f"    Packed (3 placements on 10q): {t_packed:.3f}s")
    print(f"    Packed energies:     {[f'{e:.6f}' for e in packed_energies]}")
    print(f"    Individual energies: {[f'{e:.6f}' for e in individual_energies]}")

except Exception as e:
    check("E6a.3 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E6a.4: Multi-Round on Q50 (shot-based) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver

    # Load Q50 calibration
    cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
    adapter = IQMv2Adapter()
    cal = adapter.load(cal_path)

    # Get real 4q placements
    solver = GeneralPlacementSolver()
    solver.add_device(cal)

    # 4q chain circuit (no measurements — composer adds them)
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

    # Execute first 3 rounds shot-based (Q50 composite is too large for DM)
    max_rounds = min(3, len(rounds))
    test_rounds = rounds[:max_rounds]
    total_in_rounds = sum(len(r.placements) for r in test_rounds)

    print(f"    Executing {max_rounds} rounds ({total_in_rounds} placements, shot-based)...")
    t0 = time.time()
    multi_result = execute_packed_rounds(
        circuit=qc_4q,
        observable=obs,
        rounds=test_rounds,
        device_qubits=cal.num_qubits,
        method="density_matrix",
        shots=4096,
        seed=42,
        device="CPU",
    )
    t_multi = time.time() - t0

    check("Multi-round: correct number of rounds",
          multi_result.total_rounds == max_rounds,
          f"expected {max_rounds}, got {multi_result.total_rounds}")
    check("Multi-round: all placements have results",
          multi_result.total_placements == total_in_rounds,
          f"expected {total_in_rounds}, got {multi_result.total_placements}")

    # All energies should be finite
    all_finite = all(
        pr.energy is not None and np.isfinite(pr.energy)
        for pr in multi_result.placement_results
    )
    check("Multi-round: all energies finite", all_finite)

    # No errors
    any_errors = any(pr.error is not None for pr in multi_result.placement_results)
    check("Multi-round: no placement errors", not any_errors,
          f"errors: {[pr.error for pr in multi_result.placement_results if pr.error]}")

    # All placement physical indices should be unique
    all_indices = [tuple(pr.physical_indices) for pr in multi_result.placement_results]
    check("Multi-round: all placements unique",
          len(set(all_indices)) == len(all_indices),
          f"{len(all_indices)} results, {len(set(all_indices))} unique")

    print(f"    {max_rounds} rounds, {total_in_rounds} placements in {t_multi:.2f}s")
    for rr in multi_result.rounds:
        print(f"      Round {rr.round_index}: {rr.num_placements} placements, {rr.execution_time_s:.2f}s")

    energies = [pr.energy for pr in multi_result.placement_results if pr.energy is not None]
    if energies:
        print(f"    Energy range: [{min(energies):.4f}, {max(energies):.4f}]")

except Exception as e:
    check("E6a.4 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E6a.5: Provenance — Round Index Tracking ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Each placement result should know which round it came from
    for rr in multi_result.rounds:
        for pr in rr.placement_results:
            check(f"Provenance: r{rr.round_index}_p has round_index={rr.round_index}",
                  pr.round_index == rr.round_index,
                  f"expected {rr.round_index}, got {pr.round_index}")
        # Only check first round to avoid too many checks
        break

    # Round sizes should match packing
    for i, rr in enumerate(multi_result.rounds):
        check(f"Provenance: round {i} has {len(test_rounds[i].placements)} placements",
              rr.num_placements == len(test_rounds[i].placements),
              f"expected {len(test_rounds[i].placements)}, got {rr.num_placements}")

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
        shots=0,
        seed=42,
        device="CPU",
    )

    for pr1, pr2 in zip(packed_result.placement_results,
                        repeat_result.placement_results):
        if pr1.energy is not None and pr2.energy is not None:
            diff = abs(pr1.energy - pr2.energy)
            if diff >= 1e-10:
                check(f"Determinism: placement {pr1.physical_indices}",
                      False, f"diff={diff:.2e}")
                break
    else:
        check("Determinism: all energies identical on repeat", True)

except Exception as e:
    check("E6a.6 block", False, f"Exception: {e}")
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
