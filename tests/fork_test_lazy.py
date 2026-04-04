#!/usr/bin/env python3
"""Test: lazy import inside worker vs module-level import."""
import multiprocessing as mp
import time
import os
import sys
import json

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

os.environ["OMP_NUM_THREADS"] = "1"

# Heavy parent imports — replicate sweep_engine
import h5py
import numpy as np
from lumi_hpc_qc.data.hdf5_writer import SweepHDF5Writer
from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery
from lumi_hpc_qc.sweep.noise_configs import NOISE_ENVIRONMENTS
from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
with open(cal_path) as f:
    cal_json = json.load(f)

# Trigger eigvalsh like sweep engine does
observable = SparsePauliOp.from_list([("ZZII", -1), ("IZZI", -1), ("IIZZ", -1)])
exact_e = float(np.real(np.linalg.eigvalsh(observable.to_matrix())[0]))

circuit = QuantumCircuit(4)

print(f"Parent loaded everything. PID={os.getpid()}", flush=True)

# Worker A: uses module-level run_twin_battery (EXPECTED TO HANG)
def worker_module_level(args):
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

# Worker B: lazy import inside worker (EXPECTED TO WORK)
def worker_lazy_import(args):
    i, circ, obs, cal_j = args
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

# Test lazy import first (should work)
print("\n--- Pool(8) LAZY import worker ---", flush=True)
t0 = time.time()
with mp.Pool(8) as pool:
    r = pool.map(worker_lazy_import, [(i, circuit, observable, cal_json) for i in range(8)])
print(f"Done: {time.time()-t0:.2f}s, results: {r}", flush=True)

# Test module-level import (might hang)
print("\n--- Pool(8) MODULE-LEVEL import worker ---", flush=True)
t0 = time.time()
with mp.Pool(8) as pool:
    r = pool.map(worker_module_level, [(i, circuit, observable, cal_json) for i in range(8)])
print(f"Done: {time.time()-t0:.2f}s, results: {r}", flush=True)

print("\nALL OK", flush=True)
