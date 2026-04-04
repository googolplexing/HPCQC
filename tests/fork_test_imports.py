#!/usr/bin/env python3
"""Bisect: which parent import poisons child Aer workers?"""
import multiprocessing as mp
import time
import os
import sys

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

def aer_worker(i):
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(4)
    qc.save_density_matrix()
    sim = AerSimulator(method="density_matrix", device="CPU")
    sim.run(qc, shots=0, seed_simulator=i).result()
    return i

def test_pool(label):
    print(f"  Pool(8) after {label}...", flush=True)
    t0 = time.time()
    with mp.Pool(8) as pool:
        r = pool.map(aer_worker, range(8))
    print(f"  OK: {time.time()-t0:.2f}s", flush=True)
    return True

# Baseline
print("=== 0. Baseline (no parent imports) ===", flush=True)
test_pool("nothing")

# Test each import individually
print("\n=== 1. After import numpy + eigvalsh ===", flush=True)
import numpy as np
np.linalg.eigvalsh(np.random.randn(16, 16))
test_pool("numpy+eigvalsh")

print("\n=== 2. After import h5py ===", flush=True)
import h5py
test_pool("h5py")

print("\n=== 3. After import rustworkx ===", flush=True)
import rustworkx as rx
test_pool("rustworkx")

print("\n=== 4. After import qiskit ===", flush=True)
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
test_pool("qiskit")

print("\n=== 5. After import placement_solver ===", flush=True)
from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
test_pool("placement_solver")

print("\n=== 6. After import twin_simulator ===", flush=True)
from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery
test_pool("twin_simulator")

print("\n=== 7. After import noise_configs ===", flush=True)
from lumi_hpc_qc.sweep.noise_configs import NOISE_ENVIRONMENTS
test_pool("noise_configs")

print("\n=== 8. After import hdf5_writer ===", flush=True)
from lumi_hpc_qc.data.hdf5_writer import SweepHDF5Writer
test_pool("hdf5_writer")

print("\n=== 9. After import IQMv2Adapter ===", flush=True)
from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
test_pool("IQMv2Adapter")

print("\nALL IMPORTS TESTED — ALL OK", flush=True)
