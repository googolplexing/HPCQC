#!/usr/bin/env python3
"""Test spawn context: fresh processes, no inherited C++ state."""
import multiprocessing as mp
import time
import os
import sys

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

os.environ["OMP_NUM_THREADS"] = "1"

# Heavy parent imports — same as sweep_engine
import h5py
import numpy as np
from lumi_hpc_qc.data.hdf5_writer import SweepHDF5Writer
from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery
from lumi_hpc_qc.sweep.noise_configs import NOISE_ENVIRONMENTS
from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
import json

cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
with open(cal_path) as f:
    cal_json = json.load(f)

circuit = QuantumCircuit(4)
observable = SparsePauliOp.from_list([("ZZII", -1), ("IZZI", -1), ("IIZZ", -1)])

print(f"Parent loaded everything. PID={os.getpid()}", flush=True)

def battery_worker(args):
    i, circ, obs, cal_j, proj_dir = args
    sys.path.insert(0, os.path.join(proj_dir, "src"))
    os.environ["OMP_NUM_THREADS"] = "1"
    from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery as rtb
    from lumi_hpc_qc.sweep.noise_configs import NOISE_ENVIRONMENTS as NE
    battery = rtb(
        circuit=circ, observable=obs,
        qubit_names=["QB1", "QB2", "QB3", "QB4"],
        calibration_data=cal_j,
        calibration_id="test", placement_id=f"p{i}",
        topology_hash="test_hash",
        environments=NE[:3],
        seed=42 + i, device="CPU",
    )
    return (i, len(battery.results))

# Test 1: fork (expected to hang based on previous tests)
print("\n--- fork Pool(8) ---", flush=True)
t0 = time.time()
try:
    with mp.Pool(8) as pool:
        r = pool.map(battery_worker, [(i, circuit, observable, cal_json, project_dir) for i in range(8)])
    print(f"fork OK: {time.time()-t0:.2f}s, {len(r)} results", flush=True)
except Exception as e:
    print(f"fork FAILED: {e}", flush=True)

# Test 2: spawn
print("\n--- spawn Pool(8) ---", flush=True)
t0 = time.time()
ctx = mp.get_context("spawn")
with ctx.Pool(8) as pool:
    r = pool.map(battery_worker, [(i, circuit, observable, cal_json, project_dir) for i in range(8)])
print(f"spawn OK: {time.time()-t0:.2f}s, results: {r}", flush=True)

# Test 3: spawn scaled up
print("\n--- spawn Pool(32) ---", flush=True)
t0 = time.time()
with ctx.Pool(32) as pool:
    r = pool.map(battery_worker, [(i, circuit, observable, cal_json, project_dir) for i in range(32)])
print(f"spawn OK: {time.time()-t0:.2f}s, {len(r)} results", flush=True)

print("\nALL OK", flush=True)
