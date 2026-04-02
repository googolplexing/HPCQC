import multiprocessing as mp
import time, os
import numpy as np

from qiskit.circuit.library import efficient_su2
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
print(f'Imports done. start method: {mp.get_start_method()}', flush=True)

def worker(args):
    i, seed = args
    n = 4
    terms = [("".join("Z" if k in (j,j+1) else "I" for k in range(n)), -1.0) for j in range(n-1)]
    terms += [("".join("X" if k==j else "I" for k in range(n)), -1.0) for j in range(n)]
    H = SparsePauliOp.from_list(terms)
    qc = efficient_su2(n, reps=1, entanglement="linear").decompose()
    params = np.random.default_rng(seed).uniform(-3.14, 3.14, qc.num_parameters)
    bound = qc.assign_parameters(params)
    bound.save_density_matrix()
    sim = AerSimulator(method="density_matrix", device="CPU")
    dm = np.array(sim.run(bound, shots=0).result().data()["density_matrix"])
    return float(np.real(np.trace(H.to_matrix() @ dm)))

# Test 1: Pool BEFORE any parent-process Aer usage
print("\nPool(16) BEFORE sequential...", flush=True)
t0 = time.time()
with mp.Pool(16) as pool:
    r = pool.map(worker, [(i, 42+i) for i in range(16)])
print(f"  Done in {time.time()-t0:.2f}s, {len(r)} results", flush=True)

# Test 2: Now run sequential in parent
print("\nSequential in parent (4 runs)...", flush=True)
t0 = time.time()
for i in range(4):
    e = worker((i, 42+i))
    print(f"  seq {i}: {e:.4f}", flush=True)
print(f"  Done in {time.time()-t0:.2f}s", flush=True)

# Test 3: Pool AFTER parent has used Aer — this should deadlock
print("\nPool(16) AFTER sequential (expect hang if fork-after-thread)...", flush=True)
t0 = time.time()
with mp.Pool(16) as pool:
    r = pool.map(worker, [(i, 100+i) for i in range(16)])
print(f"  Done in {time.time()-t0:.2f}s, {len(r)} results", flush=True)

print("\nALL PASSED", flush=True)
