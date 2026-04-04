#!/usr/bin/env python3
"""QPU timing + calibration data from Q50.

Submits 4 batches at different circuit depths to real Q50 hardware:
  Batch A: 10 × SU2(reps=2, 4q, 4096 shots) — campaign-depth circuits
  Batch B:  1 × SU2(reps=2, 4q, 4096 shots) — same depth, isolates overhead
  Batch C:  1 × SU2(reps=1, 4q, 4096 shots) — shallower, shows depth scaling
  Batch D:  1 × Bell pair (2q, 4096 shots)   — minimal baseline

Total: 14 circuits, 4 queue entries.

Also captures: qubit mapping, calibration set ID, coupling map,
native gate set, transpiled depths and gate counts.

Follows CSC docs:
https://docs.csc.fi/computing/quantum-computing/running-quantum-jobs/

Usage: sbatch tests/qpu_simple_timing.sh
"""
import os
import sys
import time
import json

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from iqm.qiskit_iqm import IQMProvider

# ── Connect to Q50 ──
DEVICE_CORTEX_URL = os.getenv("Q50_CORTEX_URL")
if not DEVICE_CORTEX_URL:
    raise EnvironmentError("Q50_CORTEX_URL not set — are you on q_fiqci partition?")

provider = IQMProvider(DEVICE_CORTEX_URL, quantum_computer="q50")
backend = provider.get_backend()

print("═══════════════════════════════════════════════════════")
print("  Q50 QPU TIMING + CALIBRATION")
print("═══════════════════════════════════════════════════════")
print(f"  Qubits:      {backend.num_qubits}")
print(f"  Native ops:  {backend.operation_names}")
print(f"  Coupling map ({len(backend.coupling_map)} edges):")
for edge in list(backend.coupling_map)[:10]:
    print(f"    {edge}")
if len(backend.coupling_map) > 10:
    print(f"    ... and {len(backend.coupling_map) - 10} more")
print()

SHOTS = 4096
rng = np.random.default_rng(42)

# ── Build circuits ──

# Circuit 1: SU2 reps=2, 4 qubits (campaign depth)
su2_r2 = EfficientSU2(4, reps=2, entanglement="linear").decompose()
params_r2 = rng.uniform(0, 2 * np.pi, su2_r2.num_parameters)
bound_r2 = su2_r2.assign_parameters(params_r2)
bound_r2.measure_all()
transpiled_r2 = transpile(bound_r2, backend=backend, optimization_level=2)

# Circuit 2: SU2 reps=1, 4 qubits (shallower)
su2_r1 = EfficientSU2(4, reps=1, entanglement="linear").decompose()
params_r1 = rng.uniform(0, 2 * np.pi, su2_r1.num_parameters)
bound_r1 = su2_r1.assign_parameters(params_r1)
bound_r1.measure_all()
transpiled_r1 = transpile(bound_r1, backend=backend, optimization_level=2)

# Circuit 3: Bell pair, 2 qubits (minimal)
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()
transpiled_bell = transpile(bell, backend=backend, optimization_level=2)

print("── Circuit details ──")
for name, circ in [("SU2 reps=2 4q", transpiled_r2),
                    ("SU2 reps=1 4q", transpiled_r1),
                    ("Bell pair 2q", transpiled_bell)]:
    ops = dict(circ.count_ops())
    print(f"  {name}: depth={circ.depth()}, gates={ops}")
print()


def run_batch(label, circuits, shots):
    """Submit a batch and return timing + metadata."""
    is_list = isinstance(circuits, list)
    n = len(circuits) if is_list else 1

    print(f"── {label}: {n} circuit(s), {shots} shots ──")

    t0 = time.time()
    job = backend.run(circuits, shots=shots)
    result = job.result()
    elapsed = time.time() - t0

    print(f"  Job ID:    {job.job_id()}")
    print(f"  Wall time: {elapsed * 1000:.1f} ms")

    # Get counts for first circuit
    counts = result.get_counts(0) if is_list else result.get_counts()
    top_3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
    print(f"  Top counts: {dict(top_3)}")

    # Qubit mapping
    try:
        mapping = result.request.qubit_mapping
        print(f"  Qubit mapping: {mapping}")
    except Exception as e:
        mapping = None
        print(f"  Qubit mapping: unavailable ({e})")

    # Calibration set ID
    cal_id = None
    try:
        circ_ref = circuits[0] if is_list else circuits
        exp_result = result._get_experiment(circ_ref)
        cal_id = exp_result.calibration_set_id
        print(f"  Calibration ID: {cal_id}")
    except Exception as e:
        print(f"  Calibration ID: unavailable ({e})")

    print()

    return {
        "label": label,
        "n_circuits": n,
        "shots": shots,
        "wall_time_ms": round(elapsed * 1000, 1),
        "job_id": str(job.job_id()),
        "qubit_mapping": str(mapping),
        "calibration_set_id": str(cal_id),
        "top_counts": dict(top_3),
    }


# ── Run all 4 batches ──
results = []

# Batch A: 10 × SU2 reps=2 (campaign depth)
results.append(run_batch(
    "Batch A: 10 × SU2(reps=2, 4q)",
    [transpiled_r2.copy() for _ in range(10)],
    SHOTS,
))

# Batch B: 1 × SU2 reps=2 (same depth, isolates overhead)
results.append(run_batch(
    "Batch B: 1 × SU2(reps=2, 4q)",
    transpiled_r2,
    SHOTS,
))

# Batch C: 1 × SU2 reps=1 (shallower)
results.append(run_batch(
    "Batch C: 1 × SU2(reps=1, 4q)",
    transpiled_r1,
    SHOTS,
))

# Batch D: 1 × Bell pair (minimal)
results.append(run_batch(
    "Batch D: 1 × Bell pair (2q)",
    transpiled_bell,
    SHOTS,
))

# ── Derive timing ──
t_a = results[0]["wall_time_ms"]
t_b = results[1]["wall_time_ms"]
t_c = results[2]["wall_time_ms"]
t_d = results[3]["wall_time_ms"]

per_circuit_r2 = (t_a - t_b) / 9
overhead = t_b - per_circuit_r2
depth_effect = t_b - t_c  # time difference from reps=2 vs reps=1

print("═══════════════════════════════════════════════════════")
print("  TIMING RESULTS")
print("═══════════════════════════════════════════════════════")
print(f"  Batch A (10 × SU2 reps=2):  {t_a:.1f} ms")
print(f"  Batch B (1 × SU2 reps=2):   {t_b:.1f} ms")
print(f"  Batch C (1 × SU2 reps=1):   {t_c:.1f} ms")
print(f"  Batch D (1 × Bell pair):    {t_d:.1f} ms")
print()
print(f"  Per circuit (SU2 reps=2):   {per_circuit_r2:.1f} ms")
print(f"  Batch overhead:             {overhead:.1f} ms")
print(f"  Depth effect (reps=2 vs 1): {depth_effect:.1f} ms")
print()
print("  Circuit depths (transpiled):")
print(f"    SU2 reps=2: {transpiled_r2.depth()} gates deep")
print(f"    SU2 reps=1: {transpiled_r1.depth()} gates deep")
print(f"    Bell pair:  {transpiled_bell.depth()} gates deep")
print("═══════════════════════════════════════════════════════")

# ── Save everything as JSON ──
output = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "device": "Q50",
    "num_qubits": backend.num_qubits,
    "native_ops": list(backend.operation_names),
    "shots": SHOTS,
    "circuits": {
        "su2_reps2_4q": {
            "depth": transpiled_r2.depth(),
            "gates": dict(transpiled_r2.count_ops()),
        },
        "su2_reps1_4q": {
            "depth": transpiled_r1.depth(),
            "gates": dict(transpiled_r1.count_ops()),
        },
        "bell_2q": {
            "depth": transpiled_bell.depth(),
            "gates": dict(transpiled_bell.count_ops()),
        },
    },
    "batches": results,
    "derived": {
        "per_circuit_reps2_ms": round(per_circuit_r2, 1),
        "batch_overhead_ms": round(overhead, 1),
        "depth_effect_reps2_vs_reps1_ms": round(depth_effect, 1),
    },
}

out_path = "results/qpu_simple_timing.json"
os.makedirs("results", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved: {out_path}")
