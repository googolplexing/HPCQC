#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""v1.1.1 — Batch submission ordering + DSatur packing verification.

RED-RESP-PACKING-v1.0 §3 — Mandatory blocking tests:
  Test A: Single batch ordering (result[i] == circuit[i])
  Test B: Multi-batch reassembly (correct across 200-circuit boundaries)
  Test C: DSatur produces <= greedy rounds (optimality)
  Test D: DSatur round placements are non-overlapping (correctness)

Run:
    python tests/test_v111_packing_batch.py

Expected: v1.1.1 PACKING+BATCH: ALL CHECKS PASSED
"""

import sys
import os
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
# TEST A — Single batch ordering verification
# ══════════════════════════════════════════════════════════════════════
print("\n═══ TEST A: Single batch ordering ═══")
print("  Verifies result.get_counts(i) corresponds to circuit_list[i]")

try:
    from qiskit_aer import AerSimulator

    sim = AerSimulator(method="statevector")

    # Build 5 circuits with deterministic distinct outcomes
    # Circuit i applies X gates to qubits 0..i, producing bitstring with
    # exactly i+1 ones (in different positions)
    test_circuits = []
    expected_outcomes = []
    n_qubits = 5

    for i in range(5):
        qc = QuantumCircuit(n_qubits)
        # Apply X to qubits 0 through i → produces |1...10...0>
        for q in range(i + 1):
            qc.x(q)
        qc.measure_all()
        test_circuits.append(qc)
        # Expected bitstring: qubits 0..i are 1, rest are 0
        # Qiskit bitstring is reversed: qubit 0 is rightmost
        bits = ['0'] * n_qubits
        for q in range(i + 1):
            bits[n_qubits - 1 - q] = '1'
        expected_outcomes.append(''.join(bits))

    # Submit as single batch
    result = sim.run(test_circuits, shots=100).result()

    for i in range(5):
        counts = result.get_counts(i)
        # Noiseless — should have exactly one outcome
        dominant = max(counts, key=counts.get)
        check(
            f"A.{i+1}: circuit {i} returns correct bitstring",
            dominant == expected_outcomes[i],
            f"expected {expected_outcomes[i]}, got {dominant}"
        )

    # Verify ordering is strictly preserved
    check(
        "A.6: All 5 circuits return distinct results",
        len(set(
            max(result.get_counts(i), key=result.get_counts(i).get)
            for i in range(5)
        )) == 5,
        "Some circuits returned identical results — ordering may be wrong"
    )

except Exception as e:
    check("A: Single batch test", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# TEST B — Multi-batch reassembly across chunk boundary
# ══════════════════════════════════════════════════════════════════════
print("\n═══ TEST B: Multi-batch reassembly (>200 circuits) ═══")
print("  Verifies ordering across VTT 200-circuit chunk boundaries")

try:
    from lumi_hpc_qc.backends.iqm_qpu import IqmQpuBackend

    # We can't actually call IQM without Q50 access, so we test the
    # auto-chunking logic directly. Build a mock that records call order.

    class MockIQMBackend:
        """Mock IQM backend that tracks batch sizes and returns
        deterministic results for ordering verification."""

        def __init__(self):
            self.batch_calls = []  # records (batch_size,) for each .run()
            self.total_circuits = 0

        def run(self, circuits, shots=4096):
            if isinstance(circuits, list):
                n = len(circuits)
            else:
                n = 1
                circuits = [circuits]
            self.batch_calls.append(n)
            self.total_circuits += n
            return MockResult(circuits, shots, self.total_circuits - n)

    class MockResult:
        def __init__(self, circuits, shots, offset):
            self._circuits = circuits
            self._shots = shots
            self._offset = offset  # global index of first circuit

        def result(self):
            return self

        def get_counts(self, idx=0):
            # Return a count dict encoding the global circuit index
            # so we can verify ordering
            global_idx = self._offset + idx
            # Encode global index as a bitstring
            bits = format(global_idx, '010b')
            return {bits: self._shots}

    # Create backend instance and inject mock
    backend = IqmQpuBackend.__new__(IqmQpuBackend)
    mock = MockIQMBackend()
    backend._sim = mock
    backend.VTT_BATCH_LIMIT = 200  # Use real limit

    # Submit 205 circuits — should split into [200, 5]
    dummy_circuits = [QuantumCircuit(2) for _ in range(205)]
    for qc in dummy_circuits:
        qc.measure_all()

    counts_list = backend._submit_batch(dummy_circuits, shots=1000)

    check(
        "B.1: Auto-chunking splits at 200",
        mock.batch_calls == [200, 5],
        f"expected [200, 5], got {mock.batch_calls}"
    )

    check(
        "B.2: Total results match total circuits",
        len(counts_list) == 205,
        f"expected 205, got {len(counts_list)}"
    )

    # Verify ordering: result[i] should encode global index i
    ordering_correct = True
    first_mismatch = None
    for i in range(205):
        expected_bits = format(i, '010b')
        if expected_bits not in counts_list[i]:
            ordering_correct = False
            first_mismatch = i
            break

    check(
        "B.3: Global ordering preserved across chunk boundary",
        ordering_correct,
        f"first mismatch at index {first_mismatch}" if not ordering_correct else ""
    )

    # Specifically check the boundary: index 199 (last in chunk 1)
    # and index 200 (first in chunk 2)
    check(
        "B.4: Boundary circuit 199 (last in chunk 1) correct",
        format(199, '010b') in counts_list[199],
        f"got {counts_list[199]}"
    )

    check(
        "B.5: Boundary circuit 200 (first in chunk 2) correct",
        format(200, '010b') in counts_list[200],
        f"got {counts_list[200]}"
    )

    # Test exact batch limit: 200 circuits should NOT chunk
    mock2 = MockIQMBackend()
    backend._sim = mock2
    dummy_200 = [QuantumCircuit(2) for _ in range(200)]
    for qc in dummy_200:
        qc.measure_all()
    backend._submit_batch(dummy_200, shots=1000)
    check(
        "B.6: Exactly 200 circuits → single batch (no chunking)",
        mock2.batch_calls == [200],
        f"expected [200], got {mock2.batch_calls}"
    )

    # Test 201 circuits — should chunk to [200, 1]
    mock3 = MockIQMBackend()
    backend._sim = mock3
    dummy_201 = [QuantumCircuit(2) for _ in range(201)]
    for qc in dummy_201:
        qc.measure_all()
    backend._submit_batch(dummy_201, shots=1000)
    check(
        "B.7: 201 circuits → [200, 1] chunks",
        mock3.batch_calls == [200, 1],
        f"expected [200, 1], got {mock3.batch_calls}"
    )

except Exception as e:
    check("B: Multi-batch reassembly test", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# TEST C — DSatur produces <= greedy rounds
# ══════════════════════════════════════════════════════════════════════
print("\n═══ TEST C: DSatur optimality ═══")
print("  Verifies DSatur rounds <= greedy rounds on Q50 calibration data")

try:
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IqmV2Adapter
    import json

    # Load Q50 calibration
    cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
    if os.path.exists(cal_path):
        adapter = IqmV2Adapter()
        cal = adapter.load(cal_path)

        solver = GeneralPlacementSolver()
        solver.add_device(cal)

        # Find all 4q star placements
        star_edges = [(0, 1), (0, 2), (0, 3)]
        placements = solver.find_all_placements(
            circuit_edges=star_edges,
            circuit_qubits=4,
        )

        check(
            "C.1: Found star placements on Q50",
            len(placements) > 50,
            f"found {len(placements)}, expected ~108"
        )

        # Pack with both strategies
        greedy_rounds = solver.pack_rounds(
            placements, strategy="greedy", packing_seed=42,
        )
        dsatur_rounds = solver.pack_rounds(
            placements, strategy="optimal",
        )

        check(
            "C.2: DSatur rounds <= greedy rounds",
            len(dsatur_rounds) <= len(greedy_rounds),
            f"DSatur={len(dsatur_rounds)}, greedy={len(greedy_rounds)}"
        )

        print(f"       Greedy: {len(greedy_rounds)} rounds, "
              f"DSatur: {len(dsatur_rounds)} rounds")

        # Verify all placements are assigned
        greedy_total = sum(len(r.placements) for r in greedy_rounds)
        dsatur_total = sum(len(r.placements) for r in dsatur_rounds)

        check(
            "C.3: Greedy assigns all placements",
            greedy_total == len(placements),
            f"{greedy_total} assigned vs {len(placements)} total"
        )

        check(
            "C.4: DSatur assigns all placements",
            dsatur_total == len(placements),
            f"{dsatur_total} assigned vs {len(placements)} total"
        )
    else:
        check("C: Q50 calibration file", False, f"not found at {cal_path}")

except Exception as e:
    check("C: DSatur optimality test", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# TEST D — DSatur round placements are non-overlapping
# ══════════════════════════════════════════════════════════════════════
print("\n═══ TEST D: DSatur non-overlap correctness ═══")
print("  Verifies no qubit or edge shared within any DSatur round")

try:
    if 'dsatur_rounds' in dir():
        all_ok = True
        for rnd in dsatur_rounds:
            used_q = set()
            for p in rnd.placements:
                pq = set(p.physical_indices)
                overlap = pq & used_q
                if overlap:
                    check(
                        f"D: Round {rnd.round_id} qubit overlap",
                        False,
                        f"qubits {overlap} shared"
                    )
                    all_ok = False
                    break
                used_q |= pq

        if all_ok:
            check(
                f"D.1: All {len(dsatur_rounds)} DSatur rounds have "
                f"non-overlapping qubits",
                True
            )

        # Also verify no edge overlap
        edge_ok = True
        for rnd in dsatur_rounds:
            used_e = set()
            for p in rnd.placements:
                pq = set(p.physical_indices)
                p_edges = set()
                for qi in p.physical_indices:
                    for qj in cal.adjacency.get(qi, set()):
                        if qj in pq:
                            p_edges.add((min(qi, qj), max(qi, qj)))
                overlap = p_edges & used_e
                if overlap:
                    check(
                        f"D: Round {rnd.round_id} edge overlap",
                        False,
                        f"edges {overlap} shared"
                    )
                    edge_ok = False
                    break
                used_e |= p_edges

        if edge_ok:
            check(
                f"D.2: All {len(dsatur_rounds)} DSatur rounds have "
                f"non-overlapping edges",
                True
            )
    else:
        check("D: DSatur rounds not available", False, "Test C must pass first")

except Exception as e:
    check("D: Non-overlap correctness", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
total = passed + failed
print(f"\n{'='*60}")
print(f"  v1.1.1 PACKING+BATCH: {passed}/{total} checks passed")
if failed == 0:
    print(f"  v1.1.1 PACKING+BATCH: ALL CHECKS PASSED")
else:
    print(f"  FAILURES ({failed}):")
    for e in errors:
        print(f"    - {e}")
print(f"{'='*60}")

sys.exit(0 if failed == 0 else 1)
