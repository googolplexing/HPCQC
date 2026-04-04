#!/usr/bin/env python3
"""Minimal fork test — isolate mp.Pool deadlock inside Singularity on LUMI."""
import multiprocessing as mp
import time
import os

print(f"PID={os.getpid()}, OMP={os.environ.get('OMP_NUM_THREADS', 'unset')}", flush=True)

def worker(i):
    return (i, os.getpid(), i * i)

for n in [4, 16, 64, 128]:
    t0 = time.time()
    with mp.Pool(n) as pool:
        results = pool.map(worker, range(n))
    print(f"Pool({n:>3}): {time.time()-t0:.2f}s, {len(results)} results", flush=True)

print("PURE FORK: OK", flush=True)

import numpy as np
x = np.linalg.eigvalsh(np.random.randn(16, 16))
print(f"numpy eigvalsh done", flush=True)

t0 = time.time()
with mp.Pool(64) as pool:
    results = pool.map(worker, range(64))
print(f"Pool(64) AFTER numpy: {time.time()-t0:.2f}s", flush=True)

print("ALL OK", flush=True)
