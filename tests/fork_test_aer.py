#!/usr/bin/env python3
"""Fork test with actual Aer workload — isolate where _battery_worker hangs."""
import multiprocessing as mp
import time
import os
import sys

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

print(f"PID={os.getpid()}, OMP={os.environ.get('OMP_NUM_THREADS', 'unset')}", flush=True)

def worker_aer_only(i):
    """Just import Aer and run one circuit."""
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    qc.x(i % 4)
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", device="CPU")
    result = sim.run(qc, shots=0, seed_simulator=i).result()
    return i

def worker_twin_battery(args):
    """Simulate what _battery_worker does."""
    i, cal_json = args
    from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery
    from lumi_hpc_qc.sweep.noise_configs import NOISE_ENVIRONMENTS
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(4)
    obs = SparsePauliOp.from_list([("ZZZZ", 1.0)])
    battery = run_twin_battery(
        circuit=qc, observable=obs,
        qubit_names=["QB1", "QB2", "QB3", "QB4"],
        calibration_data=cal_json,
        calibration_id="test", placement_id=f"p{i}",
        topology_hash="test_hash",
        environments=NOISE_ENVIRONMENTS[:3],  # just 3 envs for speed
        seed=42 + i, device="CPU",
    )
    return (i, len(battery.results))

# Load calibration
import json
cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
with open(cal_path) as f:
    cal_json = json.load(f)

# Test 1: Aer-only workers
print(f"\n--- Pool(16) with Aer import + run ---", flush=True)
t0 = time.time()
with mp.Pool(16) as pool:
    r = pool.map(worker_aer_only, range(16))
print(f"Done: {time.time()-t0:.2f}s, {len(r)} results", flush=True)

# Test 2: Scale up
print(f"\n--- Pool(64) with Aer ---", flush=True)
t0 = time.time()
with mp.Pool(64) as pool:
    r = pool.map(worker_aer_only, range(64))
print(f"Done: {time.time()-t0:.2f}s, {len(r)} results", flush=True)

# Test 3: Actual twin battery workers
print(f"\n--- Pool(8) with twin_battery (3 envs each) ---", flush=True)
t0 = time.time()
with mp.Pool(8) as pool:
    r = pool.map(worker_twin_battery, [(i, cal_json) for i in range(8)])
print(f"Done: {time.time()-t0:.2f}s, results: {r}", flush=True)

# Test 4: Scale twin battery
print(f"\n--- Pool(32) with twin_battery (3 envs each) ---", flush=True)
t0 = time.time()
with mp.Pool(32) as pool:
    r = pool.map(worker_twin_battery, [(i, cal_json) for i in range(32)])
print(f"Done: {time.time()-t0:.2f}s, {len(r)} results", flush=True)

print("\nALL OK", flush=True)
