#!/usr/bin/env python3
"""QPU timing benchmark for Q50 — campaign wall time calibration.

Submits 6 batches designed to isolate 4 independent variables:
  - Batch size (10 vs 1 circuit) → per-circuit execution time
  - Circuit depth (reps=2 vs reps=1) → gate execution contribution
  - Circuit width (4 qubits vs 53 qubits) → readout/reset scaling
  - Minimum baseline (2-qubit Bell pair) → absolute floor

═══════════════════════════════════════════════════════════
  BATCH DESIGN AND RATIONALE
═══════════════════════════════════════════════════════════

  NARROW CIRCUITS (4 qubits — standalone, no packing):

    Batch A: 10 × SU2(reps=2, 4q, 4096 shots)
      Purpose: Multi-circuit narrow batch. Combined with Batch B,
      isolates per-circuit execution time at campaign depth.
      Formula: per_circuit_narrow = (t_A - t_B) / 9

    Batch B: 1 × SU2(reps=2, 4q, 4096 shots)
      Purpose: Single narrow circuit at campaign depth. The difference
      from Batch A removes the per-batch overhead (queue entry,
      compilation, result collection) that both batches share.

    Batch C: 1 × SU2(reps=1, 4q, 4096 shots)
      Purpose: Shallower narrow circuit. Compared to Batch B, isolates
      the gate execution contribution of the extra reps=2 depth.
      Formula: depth_effect_narrow = t_B - t_C

    Batch D: 1 × Bell pair (2q, 4096 shots)
      Purpose: Absolute minimum circuit — 2 qubits, 2 gates. This is
      the floor: the overhead of submitting anything at all to the QPU.

  WIDE CIRCUITS (53 qubits — full chip, DSatur-equivalent width):

    Batch E: 10 × H-on-all-53-qubits + measure (4096 shots)
      Purpose: Multi-circuit full-width batch. Mirrors Batch A but at
      chip scale. Combined with Batch F, isolates per-circuit execution
      time for full-chip circuits — the actual width of DSatur composites.
      Formula: per_circuit_wide = (t_E - t_F) / 9

    Batch F: 1 × H-on-all-53-qubits + measure (4096 shots)
      Purpose: Single full-width circuit. Mirrors Batch B. This is the
      closest proxy to a single DSatur composite circuit (same width,
      minimal depth). Real composites are deeper but gate execution is
      expected to be negligible compared to readout + reset.

═══════════════════════════════════════════════════════════
  DERIVED QUANTITIES
═══════════════════════════════════════════════════════════

  per_circuit_narrow = (t_A - t_B) / 9
    Per-circuit execution time for a 4-qubit circuit (4096 shots).
    Includes: gate execution + readout (4q) + reset (4q)
    Excludes: queue wait, compilation, result transfer

  per_circuit_wide = (t_E - t_F) / 9
    Per-circuit execution time for a 53-qubit full-chip circuit.
    Includes: gate execution + readout (53q) + reset (53q)
    This is the number that matters for DSatur campaign composites.

  overhead_narrow = t_B - per_circuit_narrow
    Fixed cost per batch submission for narrow circuits.
    Includes: queue wait (should be ~0 if Q50 idle), server-side
    compilation, result serialization, network round trip.

  overhead_wide = t_F - per_circuit_wide
    Fixed cost per batch submission for full-width circuits.
    Expected to be similar to overhead_narrow (width shouldn't
    affect queue/compilation much).

  width_scaling = per_circuit_wide / per_circuit_narrow
    How much slower is a full-chip circuit vs a 4-qubit circuit?
    If readout/reset is multiplexed: scaling < 53/4.
    If readout/reset is fully parallel: scaling ≈ 1.
    If readout/reset is sequential: scaling ≈ 53/4 = 13.25.

  depth_effect = t_B - t_C
    Execution time added by the extra depth of reps=2 vs reps=1
    at 4 qubits. If this is <1ms, gate execution is negligible
    and readout+reset completely dominates.

═══════════════════════════════════════════════════════════
  CAMPAIGN WALL TIME FORMULAS (assumes no QPU queue)
═══════════════════════════════════════════════════════════

  Characterization mode (all circuits independent):
    composites = dsatur_rounds × pauli_groups × seeds
    batches = ceil(composites / 200)
    wall_time = batches × (overhead_wide + min(200, composites) × per_circuit_wide)

  VQE mode, single seed (iteration-bound):
    composites_per_iter = dsatur_rounds × pauli_groups
    wall_time = iterations × (overhead_wide + composites_per_iter × per_circuit_wide)

  VQE mode, multi-seed cross-pool (v1.2.0):
    composites_per_iter = dsatur_rounds × pauli_groups × concurrent_seeds
    batches_per_iter = ceil(composites_per_iter / 200)
    wall_time = iterations × batches_per_iter × (overhead_wide + 200 × per_circuit_wide)

  Note: These assume Q50 is idle (no queue wait). In practice, jobs
  sit in the FiQCI queue until Q50 is available. The queue wait is
  NOT per-batch — your SLURM job holds the QPU for its entire
  duration. Queue wait is a one-time cost at job start, then all
  batches execute without additional queuing.

═══════════════════════════════════════════════════════════

Total: 24 circuits, 6 queue entries, 98,304 shots.

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

# Get coupling map edges safely (CouplingMap doesn't support len())
try:
    coupling_edges = backend.coupling_map.get_edges()
    n_edges = len(coupling_edges)
except Exception:
    coupling_edges = list(backend.coupling_map)
    n_edges = len(coupling_edges)

NUM_QUBITS = backend.num_qubits  # should be 53

print("═══════════════════════════════════════════════════════")
print("  Q50 QPU TIMING BENCHMARK")
print("═══════════════════════════════════════════════════════")
print(f"  Qubits:      {NUM_QUBITS}")
print(f"  Native ops:  {backend.operation_names}")
print(f"  Coupling map: {n_edges} edges")
print()

SHOTS = 4096
rng = np.random.default_rng(42)

# ═══════════════════════════════════════════════════════
# BUILD CIRCUITS
# ═══════════════════════════════════════════════════════

# ── Narrow: SU2 reps=2, 4 qubits (campaign depth) ──
su2_r2 = EfficientSU2(4, reps=2, entanglement="linear").decompose()
params_r2 = rng.uniform(0, 2 * np.pi, su2_r2.num_parameters)
bound_r2 = su2_r2.assign_parameters(params_r2)
bound_r2.measure_all()
transpiled_r2 = transpile(bound_r2, backend=backend, optimization_level=2)

# ── Narrow: SU2 reps=1, 4 qubits (shallower) ──
su2_r1 = EfficientSU2(4, reps=1, entanglement="linear").decompose()
params_r1 = rng.uniform(0, 2 * np.pi, su2_r1.num_parameters)
bound_r1 = su2_r1.assign_parameters(params_r1)
bound_r1.measure_all()
transpiled_r1 = transpile(bound_r1, backend=backend, optimization_level=2)

# ── Narrow: Bell pair, 2 qubits (minimal) ──
bell = QuantumCircuit(2)
bell.h(0)
bell.cx(0, 1)
bell.measure_all()
transpiled_bell = transpile(bell, backend=backend, optimization_level=2)

# ── Wide: H on all 53 qubits + measure (full chip, minimal depth) ──
wide = QuantumCircuit(NUM_QUBITS)
for i in range(NUM_QUBITS):
    wide.h(i)
wide.measure_all()
transpiled_wide = transpile(wide, backend=backend, optimization_level=2)

print("── Circuit details (after transpilation) ──")
for name, circ in [("SU2 reps=2 4q (narrow deep)", transpiled_r2),
                    ("SU2 reps=1 4q (narrow shallow)", transpiled_r1),
                    ("Bell pair 2q (minimal)", transpiled_bell),
                    (f"H-all {NUM_QUBITS}q (wide shallow)", transpiled_wide)]:
    ops = dict(circ.count_ops())
    print(f"  {name}: depth={circ.depth()}, qubits={circ.num_qubits}, gates={ops}")
print()

# Safety check: confirm circuit counts before submission
EXPECTED_TOTAL = 24  # 10 + 1 + 1 + 1 + 10 + 1
print(f"── Pre-flight safety check ──")
print(f"  Total circuits to submit: {EXPECTED_TOTAL}")
print(f"  Total queue entries: 6")
print(f"  Total shots: {EXPECTED_TOTAL * SHOTS:,}")
print()


# ═══════════════════════════════════════════════════════
# BATCH EXECUTION
# ═══════════════════════════════════════════════════════

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
    n_unique = len(counts)
    top_3 = sorted(counts.items(), key=lambda x: -x[1])[:3]
    print(f"  Unique bitstrings: {n_unique}")
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
        "per_circuit_ms": round(elapsed * 1000 / n, 1),
        "job_id": str(job.job_id()),
        "qubit_mapping": str(mapping),
        "calibration_set_id": str(cal_id),
        "top_counts": dict(top_3),
        "unique_bitstrings": n_unique,
    }


# ── Run all 6 batches ──
results = []

# Batch A: 10 × SU2 reps=2 (narrow, campaign depth)
results.append(run_batch(
    "Batch A: 10 × SU2(reps=2, 4q) [narrow multi]",
    [transpiled_r2.copy() for _ in range(10)],
    SHOTS,
))

# Batch B: 1 × SU2 reps=2 (narrow, campaign depth, single)
results.append(run_batch(
    "Batch B: 1 × SU2(reps=2, 4q) [narrow single]",
    transpiled_r2,
    SHOTS,
))

# Batch C: 1 × SU2 reps=1 (narrow, shallow)
results.append(run_batch(
    "Batch C: 1 × SU2(reps=1, 4q) [narrow shallow]",
    transpiled_r1,
    SHOTS,
))

# Batch D: 1 × Bell pair (minimal)
results.append(run_batch(
    "Batch D: 1 × Bell(2q) [minimal baseline]",
    transpiled_bell,
    SHOTS,
))

# Batch E: 10 × H-all-53q (wide, multi)
results.append(run_batch(
    f"Batch E: 10 × H-all({NUM_QUBITS}q) [wide multi]",
    [transpiled_wide.copy() for _ in range(10)],
    SHOTS,
))

# Batch F: 1 × H-all-53q (wide, single)
results.append(run_batch(
    f"Batch F: 1 × H-all({NUM_QUBITS}q) [wide single]",
    transpiled_wide,
    SHOTS,
))


# ═══════════════════════════════════════════════════════
# DERIVE TIMING
# ═══════════════════════════════════════════════════════

t_a = results[0]["wall_time_ms"]
t_b = results[1]["wall_time_ms"]
t_c = results[2]["wall_time_ms"]
t_d = results[3]["wall_time_ms"]
t_e = results[4]["wall_time_ms"]
t_f = results[5]["wall_time_ms"]

# Per-circuit execution time (subtract single from batch-of-10, divide by 9)
per_circuit_narrow = (t_a - t_b) / 9
per_circuit_wide = (t_e - t_f) / 9

# Per-batch overhead (single circuit time minus per-circuit time)
overhead_narrow = t_b - per_circuit_narrow
overhead_wide = t_f - per_circuit_wide

# Width scaling: how much more does full-chip cost vs 4 qubits?
width_scaling = per_circuit_wide / per_circuit_narrow if per_circuit_narrow > 0 else float("nan")

# Depth effect: extra time from reps=2 vs reps=1 at 4 qubits
depth_effect = t_b - t_c

# ── Campaign projections (assumes no QPU queue) ──
# TFIM 4q star, 108 placements, DSatur 16 rounds, 2 Pauli groups

# Characterization mode: all circuits independent
def project_characterization(seeds, rounds=16, groups=2):
    composites = rounds * groups * seeds
    batches = -(-composites // 200)  # ceil division
    circuits_last = composites % 200 or 200
    wall = (batches - 1) * (overhead_wide + 200 * per_circuit_wide) + \
           (overhead_wide + circuits_last * per_circuit_wide)
    return composites, batches, wall

# VQE mode: iteration-bound, single seed
def project_vqe_single(iterations=400, rounds=16, groups=2):
    composites_per_iter = rounds * groups  # 32
    wall_per_iter = overhead_wide + composites_per_iter * per_circuit_wide
    return composites_per_iter, wall_per_iter, iterations * wall_per_iter

# VQE mode: cross-seed pool, N seeds sharing batches
def project_vqe_pool(seeds, iterations=400, rounds=16, groups=2):
    composites_per_iter = rounds * groups * seeds
    batches_per_iter = -(-composites_per_iter // 200)
    wall_per_iter = batches_per_iter * (overhead_wide + min(200, composites_per_iter) * per_circuit_wide)
    return composites_per_iter, batches_per_iter, iterations * wall_per_iter


print("═══════════════════════════════════════════════════════")
print("  TIMING RESULTS")
print("═══════════════════════════════════════════════════════")
print()
print("  Raw batch times:")
print(f"    Batch A (10 × 4q deep):    {t_a:,.1f} ms")
print(f"    Batch B (1 × 4q deep):     {t_b:,.1f} ms")
print(f"    Batch C (1 × 4q shallow):  {t_c:,.1f} ms")
print(f"    Batch D (1 × 2q minimal):  {t_d:,.1f} ms")
print(f"    Batch E (10 × 53q wide):   {t_e:,.1f} ms")
print(f"    Batch F (1 × 53q wide):    {t_f:,.1f} ms")
print()
print("  Derived (narrow — 4 qubits):")
print(f"    Per circuit:     {per_circuit_narrow:,.1f} ms")
print(f"    Batch overhead:  {overhead_narrow:,.1f} ms")
print()
print("  Derived (wide — 53 qubits, DSatur-equivalent):")
print(f"    Per circuit:     {per_circuit_wide:,.1f} ms")
print(f"    Batch overhead:  {overhead_wide:,.1f} ms")
print()
print(f"  Width scaling:     {width_scaling:.2f}× (53q vs 4q per circuit)")
print(f"  Depth effect:      {depth_effect:,.1f} ms (reps=2 vs reps=1)")
print()
print("  Transpiled depths:")
print(f"    SU2 reps=2 4q:  {transpiled_r2.depth()} layers")
print(f"    SU2 reps=1 4q:  {transpiled_r1.depth()} layers")
print(f"    Bell 2q:         {transpiled_bell.depth()} layers")
print(f"    H-all 53q:       {transpiled_wide.depth()} layers")

# ── Campaign projections ──
print()
print("═══════════════════════════════════════════════════════")
print("  CAMPAIGN WALL TIME PROJECTIONS (no QPU queue)")
print("═══════════════════════════════════════════════════════")
print()

# Characterization
for seeds in [1, 20, 100]:
    comp, batches, wall = project_characterization(seeds)
    print(f"  Characterization ({seeds} seed{'s' if seeds>1 else ''}):")
    print(f"    {comp} composites → {batches} batch{'es' if batches>1 else ''} → {wall/1000:,.1f} sec ({wall/60000:,.1f} min)")

print()

# VQE single seed
comp_iter, wall_iter, wall_total = project_vqe_single()
print(f"  VQE single seed (400 iterations):")
print(f"    {comp_iter} composites/iter → {wall_iter/1000:,.1f} sec/iter → {wall_total/60000:,.1f} min total")

print()

# VQE cross-seed pool
for seeds in [6, 20]:
    comp_iter, batches_iter, wall_total = project_vqe_pool(seeds)
    print(f"  VQE {seeds}-seed pool (400 iterations):")
    print(f"    {comp_iter} composites/iter → {batches_iter} batch{'es' if batches_iter>1 else ''}/iter → {wall_total/60000:,.1f} min total")

print()
print("═══════════════════════════════════════════════════════")


# ═══════════════════════════════════════════════════════
# SAVE RESULTS
# ═══════════════════════════════════════════════════════

output = {
    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    "device": "Q50",
    "num_qubits": NUM_QUBITS,
    "native_ops": list(backend.operation_names),
    "coupling_map_edges": n_edges,
    "shots": SHOTS,
    "total_circuits": EXPECTED_TOTAL,
    "total_queue_entries": 6,
    "circuits": {
        "su2_reps2_4q": {
            "width": 4,
            "depth": transpiled_r2.depth(),
            "gates": dict(transpiled_r2.count_ops()),
            "category": "narrow_deep",
        },
        "su2_reps1_4q": {
            "width": 4,
            "depth": transpiled_r1.depth(),
            "gates": dict(transpiled_r1.count_ops()),
            "category": "narrow_shallow",
        },
        "bell_2q": {
            "width": 2,
            "depth": transpiled_bell.depth(),
            "gates": dict(transpiled_bell.count_ops()),
            "category": "minimal",
        },
        f"h_all_{NUM_QUBITS}q": {
            "width": NUM_QUBITS,
            "depth": transpiled_wide.depth(),
            "gates": dict(transpiled_wide.count_ops()),
            "category": "wide_shallow",
        },
    },
    "batches": results,
    "derived": {
        "per_circuit_narrow_ms": round(per_circuit_narrow, 1),
        "per_circuit_wide_ms": round(per_circuit_wide, 1),
        "overhead_narrow_ms": round(overhead_narrow, 1),
        "overhead_wide_ms": round(overhead_wide, 1),
        "width_scaling_factor": round(width_scaling, 3),
        "depth_effect_ms": round(depth_effect, 1),
    },
    "campaign_projections": {
        "characterization_1_seed": {
            "composites": project_characterization(1)[0],
            "batches": project_characterization(1)[1],
            "wall_time_ms": round(project_characterization(1)[2], 1),
        },
        "characterization_20_seeds": {
            "composites": project_characterization(20)[0],
            "batches": project_characterization(20)[1],
            "wall_time_ms": round(project_characterization(20)[2], 1),
        },
        "vqe_single_seed_400iter": {
            "composites_per_iter": project_vqe_single()[0],
            "wall_per_iter_ms": round(project_vqe_single()[1], 1),
            "wall_total_ms": round(project_vqe_single()[2], 1),
        },
        "vqe_6seed_pool_400iter": {
            "composites_per_iter": project_vqe_pool(6)[0],
            "batches_per_iter": project_vqe_pool(6)[1],
            "wall_total_ms": round(project_vqe_pool(6)[2], 1),
        },
    },
    "formulas": {
        "per_circuit_narrow": "(t_A - t_B) / 9",
        "per_circuit_wide": "(t_E - t_F) / 9",
        "overhead_narrow": "t_B - per_circuit_narrow",
        "overhead_wide": "t_F - per_circuit_wide",
        "width_scaling": "per_circuit_wide / per_circuit_narrow",
        "depth_effect": "t_B - t_C",
        "characterization_wall": "ceil(composites/200) × (overhead_wide + min(200,N) × per_circuit_wide)",
        "vqe_single_wall": "iterations × (overhead_wide + composites_per_iter × per_circuit_wide)",
        "vqe_pool_wall": "iterations × ceil(composites_per_iter/200) × (overhead_wide + 200 × per_circuit_wide)",
    },
}

out_path = "results/qpu_simple_timing.json"
os.makedirs("results", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(output, f, indent=2)
print(f"Results saved: {out_path}")
