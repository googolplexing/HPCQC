#!/usr/bin/env python3
"""Replicate EXACTLY what sweep_engine._execute_group does before Pool."""
import multiprocessing as mp
import time
import os
import sys
import json
import tempfile

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

os.environ["OMP_NUM_THREADS"] = "1"

print(f"PID={os.getpid()}", flush=True)

# ── Step 1: Import everything sweep_engine imports at module level ──
print("Step 1: Module imports...", flush=True)
import h5py
import numpy as np
from lumi_hpc_qc.data.hdf5_writer import SweepHDF5Writer
from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery
from lumi_hpc_qc.sweep.noise_configs import NOISE_ENVIRONMENTS
from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
print("  Done", flush=True)

# ── Step 2: Load calibration (like SweepEngine._load_calibration) ──
print("Step 2: Load calibration...", flush=True)
cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
adapter = IQMv2Adapter()
cal = adapter.load(cal_path)
with open(cal_path) as f:
    cal_json = json.load(f)
print(f"  Device: {cal.device_id}, {cal.num_qubits} qubits", flush=True)

# ── Step 3: Build circuit + observable (like _build_circuit_and_observable) ──
print("Step 3: Build circuit + exact_ground_energy (triggers eigvalsh)...", flush=True)
circuit = QuantumCircuit(4)
observable = SparsePauliOp.from_list([("ZZII", -1), ("IZZI", -1), ("IIZZ", -1),
                                      ("XIII", -1), ("IXII", -1), ("IIXI", -1), ("IIIX", -1)])
exact_e = float(np.real(np.linalg.eigvalsh(observable.to_matrix())[0]))
print(f"  exact_ground_energy = {exact_e:.4f}", flush=True)

# ── Step 4: Open HDF5 writer (like SweepEngine.run context manager) ──
print("Step 4: Open HDF5 writer...", flush=True)
tmp_h5 = os.path.join(tempfile.mkdtemp(), "test_sweep.h5")
writer = SweepHDF5Writer(tmp_h5, sweep_attrs={"sweep_id": "fork_test"})
writer.open()
print(f"  HDF5 open at {tmp_h5}", flush=True)

# ── Step 5: Pool WITHOUT closing h5py first ──
def simple_worker(i):
    return (i, os.getpid())

print("\nStep 5a: Pool(16) with h5py OPEN...", flush=True)
t0 = time.time()
with mp.Pool(16) as pool:
    r = pool.map(simple_worker, range(16))
print(f"  Done: {time.time()-t0:.2f}s, {len(r)} results", flush=True)

# ── Step 6: Close h5py, then Pool ──
print("\nStep 5b: Close h5py, then Pool(16)...", flush=True)
writer.close()
t0 = time.time()
with mp.Pool(16) as pool:
    r = pool.map(simple_worker, range(16))
print(f"  Done: {time.time()-t0:.2f}s, {len(r)} results", flush=True)

# ── Step 7: Test with actual battery worker ──
print("\nStep 6: Pool(8) with battery worker, h5py closed...", flush=True)

def battery_worker(args):
    i, circ, obs, cal_j = args
    battery = run_twin_battery(
        circuit=circ, observable=obs,
        qubit_names=["QB1", "QB2", "QB3", "QB4"],
        calibration_data=cal_j,
        calibration_id="test", placement_id=f"p{i}",
        topology_hash="test_hash",
        environments=NOISE_ENVIRONMENTS[:3],
        seed=42 + i, device="CPU",
    )
    return (i, len(battery.results))

t0 = time.time()
with mp.Pool(8) as pool:
    r = pool.map(battery_worker, [(i, circuit, observable, cal_json) for i in range(8)])
print(f"  Done: {time.time()-t0:.2f}s, results: {r}", flush=True)

# ── Step 8: Reopen h5py, test battery worker again ──
print("\nStep 7: Reopen h5py, then Pool(8) with battery worker...", flush=True)
writer.open()
writer.close()  # close before pool
t0 = time.time()
with mp.Pool(8) as pool:
    r = pool.map(battery_worker, [(i, circuit, observable, cal_json) for i in range(8)])
print(f"  Done: {time.time()-t0:.2f}s, results: {r}", flush=True)
writer.open()  # reopen for cleanup
writer.close()

print("\nALL OK", flush=True)
