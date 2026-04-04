#!/usr/bin/env python3
"""Test: run Pool in a clean subprocess to avoid inherited C++ state."""
import os
import sys
import time
import json
import pickle
import subprocess
import tempfile

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

# Build work items exactly like sweep engine
cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
with open(cal_path) as f:
    cal_json = json.load(f)

circuit = QuantumCircuit(4)
observable = SparsePauliOp.from_list([("ZZII", -1), ("IZZI", -1), ("IIZZ", -1)])
exact_e = float(np.real(np.linalg.eigvalsh(observable.to_matrix())[0]))

noise_envs = NOISE_ENVIRONMENTS[:3]

work_items = []
for i in range(100):
    work_items.append((
        circuit, observable,
        ["QB1", "QB2", "QB3", "QB4"],
        cal_json, "test_cal",
        f"p{i}", "test_hash",
        noise_envs,
        42 + i, "CPU", {},
    ))

print(f"Parent PID={os.getpid()}, {len(work_items)} work items built", flush=True)

# Serialize work items to temp file
tmp_dir = tempfile.mkdtemp()
items_path = os.path.join(tmp_dir, "work_items.pkl")
results_path = os.path.join(tmp_dir, "results.pkl")

with open(items_path, "wb") as f:
    pickle.dump(work_items, f)
print(f"Serialized {len(work_items)} items ({os.path.getsize(items_path) / 1024:.0f} KB)", flush=True)

# Run Pool in a CLEAN subprocess
worker_script = os.path.join(tmp_dir, "pool_runner.py")
with open(worker_script, "w") as f:
    f.write(f'''
import sys, os, pickle, time, multiprocessing as mp
sys.path.insert(0, os.path.join("{project_dir}", "src"))
os.environ["OMP_NUM_THREADS"] = "1"

from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery

def battery_worker(args):
    (circuit, observable, qubit_names, cal_json, cal_id,
     placement_id, topology_hash, noise_envs, seed,
     device, noiseless_cache) = args
    try:
        battery = run_twin_battery(
            circuit=circuit, observable=observable,
            qubit_names=qubit_names, calibration_data=cal_json,
            calibration_id=cal_id, placement_id=placement_id,
            topology_hash=topology_hash, environments=noise_envs,
            seed=seed, device=device, noiseless_cache=noiseless_cache,
        )
        return {{"placement_id_str": placement_id, "battery": battery, "error": None}}
    except Exception as e:
        return {{"placement_id_str": placement_id, "battery": None, "error": str(e)}}

with open("{items_path}", "rb") as f:
    work_items = pickle.load(f)

n = len(work_items)
workers = min(100, n)
print(f"Subprocess PID={{os.getpid()}}, Pool({{workers}}) for {{n}} items", flush=True)

t0 = time.time()
with mp.Pool(workers) as pool:
    results = pool.map(battery_worker, work_items)
elapsed = time.time() - t0
print(f"Pool complete: {{elapsed:.2f}}s, {{len(results)}} results", flush=True)

ok = sum(1 for r in results if r["error"] is None)
print(f"  Success: {{ok}}/{{len(results)}}", flush=True)

with open("{results_path}", "wb") as f:
    pickle.dump(results, f)
''')

print(f"\nLaunching clean subprocess with Pool(100)...", flush=True)
t0 = time.time()
result = subprocess.run(
    [sys.executable, worker_script],
    capture_output=False,
)
elapsed = time.time() - t0
print(f"Subprocess returned: {result.returncode}, {elapsed:.2f}s", flush=True)

# Read results back
with open(results_path, "rb") as f:
    results = pickle.load(f)
ok = sum(1 for r in results if r["error"] is None)
print(f"Results: {ok}/{len(results)} successful", flush=True)

print(f"\nALL OK", flush=True)
