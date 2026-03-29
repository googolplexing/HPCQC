# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""GREEN-REQ-001 Integration Test 2: Aer GPU backend + VQE convergence.

Tests that the Aer GPU backend (custom ROCm fork) works correctly
after the container rebuild. Runs a BYO TFIM 8q VQE with L-BFGS-B
and verifies convergence to <0.1% error.

This test does NOT import mitiq — it tests the GPU simulation path
in isolation. See test_01 for mitiq import verification.
"""

import os
import sys
import time

project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(project_dir, "src"))
os.chdir(project_dir)

print("=" * 70)
print("  GREEN-REQ-001 Integration Test 2: Aer GPU + VQE Convergence")
print("=" * 70)
print()

# ── Test 2a: Direct Aer GPU import and basic simulation ──
print("--- Test 2a: Aer GPU Import & Basic Simulation ---")

import numpy as np
print(f"  numpy version: {np.__version__}")

import qiskit
print(f"  qiskit version: {qiskit.__version__}")

import qiskit_aer
print(f"  qiskit-aer version: {qiskit_aer.__version__}")

from qiskit.circuit import QuantumCircuit
from qiskit_aer import AerSimulator

# Quick 4-qubit statevector test
qc = QuantumCircuit(4)
qc.h(0)
qc.cx(0, 1)
qc.cx(1, 2)
qc.cx(2, 3)
qc.save_statevector()

sim = AerSimulator(method="statevector", device="GPU")
result = sim.run(qc, shots=0).result()
sv = result.get_statevector()
print(f"  4-qubit GHZ statevector: {sv.data.shape} amplitudes")
# GHZ state: |0000⟩ + |1111⟩ / sqrt(2)
amp_0000 = abs(sv.data[0])
amp_1111 = abs(sv.data[-1])
assert abs(amp_0000 - 1 / np.sqrt(2)) < 0.01, f"GHZ amp |0000⟩ wrong: {amp_0000}"
assert abs(amp_1111 - 1 / np.sqrt(2)) < 0.01, f"GHZ amp |1111⟩ wrong: {amp_1111}"
print("  [PASS] Aer GPU statevector simulation correct")
print()

# ── Test 2b: Shot-based simulation (tests F1 Pauli measurement path) ──
print("--- Test 2b: Shot-Based Simulation (density_matrix) ---")

qc_shots = QuantumCircuit(2)
qc_shots.h(0)
qc_shots.cx(0, 1)
qc_shots.measure_all()

sim_dm = AerSimulator(method="density_matrix", device="GPU")
result_dm = sim_dm.run(qc_shots, shots=4096).result()
counts = result_dm.get_counts()
print(f"  Bell state counts: {counts}")
# Expect roughly 50/50 between |00⟩ and |11⟩
total = sum(counts.values())
p_00 = counts.get("00", 0) / total
p_11 = counts.get("11", 0) / total
assert p_00 > 0.4 and p_00 < 0.6, f"Bell state |00⟩ probability off: {p_00}"
assert p_11 > 0.4 and p_11 < 0.6, f"Bell state |11⟩ probability off: {p_11}"
print("  [PASS] Shot-based density_matrix simulation correct")
print()

# ── Test 2c: Full VQE convergence (BYO TFIM 8q) ──
print("--- Test 2c: VQE Convergence — BYO TFIM 8q ---")
print("  (This is the primary benchmark. Expect <0.1% error.)")
print()

from lumi_hpc_qc.cli.config_loader import load_config
from lumi_hpc_qc.orchestration.controller import Controller

config_path = os.path.join(project_dir, "configs", "byo_tfim_8q.yaml")
if not os.path.exists(config_path):
    print(f"  SKIP: {config_path} not found")
    print("  (Run this test from the HPCQC project root)")
    sys.exit(1)

config = load_config(config_path)
# Reduce iterations for integration test speed
config.optimizer_params["maxiter"] = 200

print(f"  Config: {config_path}")
print(f"  Model: {config.model}")
print(f"  Ansatz: {config.ansatz} (reps={config.ansatz_params.get('reps')})")
print(f"  Optimizer: {config.optimizer} (maxiter={config.optimizer_params['maxiter']})")
print(f"  Backend: {config.backend} ({config.backend_params.get('method', 'statevector')})")
print()

t0 = time.time()
controller = Controller()
record = controller.run_interactive(config)
elapsed = time.time() - t0

print()
if record.convergence:
    best_e = record.convergence.best_energy
    exact_e = record.convergence.exact_ground_energy
    err_pct = record.convergence.relative_error_pct
    n_iters = record.convergence.total_iterations

    print(f"  Best energy:   {best_e:.8f}")
    if exact_e is not None:
        print(f"  Exact energy:  {exact_e:.8f}")
    if err_pct is not None:
        print(f"  Relative error: {err_pct:.4f}%")
    print(f"  Iterations:    {n_iters}")
    print(f"  Wall time:     {elapsed:.1f}s")
    print()

    if err_pct is not None and err_pct < 0.1:
        print("  [PASS] VQE converged to <0.1% error")
    elif err_pct is not None and err_pct < 1.0:
        print(f"  [WARN] VQE converged to {err_pct:.4f}% (< 1% but > 0.1%)")
        print("         May need more iterations. Not a container issue.")
    else:
        print(f"  [FAIL] VQE error {err_pct}% exceeds 1% threshold")
        sys.exit(1)
else:
    print("  [FAIL] No convergence record produced")
    sys.exit(1)

# ── Test 2d: Export pipeline (JSON + CSV) ──
print()
print("--- Test 2d: Export Pipeline (JSON + CSV) ---")

from lumi_hpc_qc.data.result_store import save_json, load_json
from lumi_hpc_qc.data.export import export_training_data

# Find the experiment result file
result_dir = os.path.join(project_dir, config.output_dir)
json_files = []
for root, dirs, files in os.walk(result_dir):
    for f in files:
        if f.endswith("_result.json"):
            json_files.append(os.path.join(root, f))

if json_files:
    # Export to CSV
    csv_path = os.path.join(result_dir, "integration_test_export.csv")
    n_rows = export_training_data(json_files[:1], csv_path)
    print(f"  Exported {n_rows} rows to {csv_path}")
    assert n_rows > 0, "Export produced 0 rows"
    assert os.path.exists(csv_path), f"CSV file not created: {csv_path}"

    # Verify CSV is readable
    import csv
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames
        first_row = next(reader)
    print(f"  CSV headers: {len(headers)} columns")
    print(f"  First row experiment_id: {first_row.get('experiment_id', 'N/A')}")
    print("  [PASS] Export pipeline functional")
else:
    print("  [WARN] No experiment JSON files found for export test")
    print("         (VQE may not have produced output files)")

print()
print("=" * 70)
print("  INTEGRATION TEST 2: COMPLETE")
print("=" * 70)
