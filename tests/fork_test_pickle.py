#!/usr/bin/env python3
"""Test: does pickling QuantumCircuit/SparsePauliOp through pool.map hang?"""
import multiprocessing as mp
import time
import os
import sys

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

os.environ["OMP_NUM_THREADS"] = "1"

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

print(f"PID={os.getpid()}", flush=True)

# Build circuit + observable in PARENT (like sweep_engine does)
circuit = QuantumCircuit(4)
observable = SparsePauliOp.from_list([("ZZZZ", 1.0), ("XXXX", -0.5)])

def worker_receives_objects(args):
    """Worker receives pre-built circuit/observable via pickle."""
    i, circ, obs = args
    from qiskit_aer import AerSimulator
    sim = AerSimulator(method="density_matrix", device="CPU")
    circ_copy = circ.copy()
    circ_copy.save_density_matrix()
    result = sim.run(circ_copy, shots=0, seed_simulator=i).result()
    return i

def worker_builds_own(i):
    """Worker builds its own circuit/observable (like fork_test_aer)."""
    from qiskit_aer import AerSimulator
    from qiskit import QuantumCircuit as QC
    from qiskit.quantum_info import SparsePauliOp as SPO
    circ = QC(4)
    circ.save_density_matrix()
    sim = AerSimulator(method="density_matrix", device="CPU")
    result = sim.run(circ, shots=0, seed_simulator=i).result()
    return i

# Test 1: Workers build their own (should work — like fork_test_aer)
print("\n--- Pool(16) workers build own circuit ---", flush=True)
t0 = time.time()
with mp.Pool(16) as pool:
    r = pool.map(worker_builds_own, range(16))
print(f"Done: {time.time()-t0:.2f}s, {len(r)} results", flush=True)

# Test 2: Pass circuit through pool.map (like sweep_engine does)
print("\n--- Pool(16) pass circuit via pickle ---", flush=True)
t0 = time.time()
with mp.Pool(16) as pool:
    r = pool.map(worker_receives_objects, [(i, circuit, observable) for i in range(16)])
print(f"Done: {time.time()-t0:.2f}s, {len(r)} results", flush=True)

# Test 3: Scale up
print("\n--- Pool(64) pass circuit via pickle ---", flush=True)
t0 = time.time()
with mp.Pool(64) as pool:
    r = pool.map(worker_receives_objects, [(i, circuit, observable) for i in range(64)])
print(f"Done: {time.time()-t0:.2f}s, {len(r)} results", flush=True)

print("\nALL OK", flush=True)
