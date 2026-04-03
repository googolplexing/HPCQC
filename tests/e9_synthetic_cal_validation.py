#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E — E9: Synthetic Calibration Tools Validation.

Tests the CLI tool and programmatic API for generating perturbed
calibration files from real Q50 calibration data.

Validates:
  - All perturbation types (scale_t1/t2/readout/gate, poison, uniform, improve)
  - Physical constraints enforced (T2 ≤ 2*T1, readout clamped)
  - Provenance metadata (_synthetic_metadata) in output JSON
  - Output is valid IQM v2 format (round-trip loadable)
  - Batch generation across multiple noise regimes
  - Synthetic calibration produces different twin sim results vs real
  - Key validation rejects unknown perturbation keys

Run on LUMI standard partition:
    srun ... python tests/e9_synthetic_cal_validation.py

Expected: E9 VALIDATION: ALL CHECKS PASSED

RED-SPEC-002 §10
"""

import sys
import os
import json
import time
import tempfile
import traceback

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        failed += 1
        errors.append(f"{name}: {detail}")


# Find calibration file
cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
if not os.path.exists(cal_path):
    cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260326.json")
assert os.path.exists(cal_path), f"No calibration file found in {project_dir}/examples/"

test_dir = tempfile.mkdtemp(prefix="e9_test_")


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E9.1: Programmatic API — generate_synthetic ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.data.tools.perturb_calibration import (
        generate_synthetic, generate_batch,
    )

    # Basic T1 scaling
    out_path = os.path.join(test_dir, "synth_t1_07.json")
    result = generate_synthetic(cal_path, {"scale_t1": 0.7}, out_path)

    check("generate_synthetic returns result dict",
          isinstance(result, dict) and "num_qubits" in result)
    check("Output file created", os.path.exists(out_path))
    check("Is synthetic", result["is_synthetic"])
    check("53 qubits in output", result["num_qubits"] == 53,
          f"got {result['num_qubits']}")
    check("Has gates", result["num_gates"] > 0)
    check("Description contains perturbation",
          "scale_t1" in result["description"])

except Exception as e:
    check("E9.1 programmatic API", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E9.2: Output Format — Valid IQM v2 JSON ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    with open(out_path) as f:
        synth_json = json.load(f)

    check("Output has calibration_set_id",
          "calibration_set_id" in synth_json)
    check("Output has timestamp", "timestamp" in synth_json)
    check("Output has device", "device" in synth_json)
    check("Output has qubits", "qubits" in synth_json)
    check("Output has two_qubit_gates", "two_qubit_gates" in synth_json)

    # Check qubit structure
    sample_qubit = list(synth_json["qubits"].values())[0]
    check("Qubit has t1_us", "t1_us" in sample_qubit)
    check("Qubit has t2_us", "t2_us" in sample_qubit)
    check("Qubit has readout_fidelity", "readout_fidelity" in sample_qubit)
    check("Qubit has single_gate_error", "single_gate_error" in sample_qubit)

    # Verify round-trip: load synthetic with IQM adapter
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    adapter = IQMv2Adapter()
    loaded = adapter.load(out_path)
    check("Round-trip: loadable by IQMv2Adapter",
          loaded.num_qubits == 53)

except Exception as e:
    check("E9.2 output format", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E9.3: Provenance Metadata ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    check("Output has _synthetic_metadata",
          "_synthetic_metadata" in synth_json)

    meta = synth_json["_synthetic_metadata"]
    check("Provenance has source_file", "source_file" in meta)
    check("Provenance has perturbation", "perturbation" in meta)
    check("Provenance has generation_timestamp", "generation_timestamp" in meta)
    check("Provenance has generator_version", "generator_version" in meta)
    check("Provenance has is_synthetic=True", meta.get("is_synthetic") is True)
    check("Provenance perturbation mentions scale_t1",
          "scale_t1" in str(meta.get("perturbation", "")))

except Exception as e:
    check("E9.3 provenance", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E9.4: T1 Scaling Correctness ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Load original calibration
    with open(cal_path) as f:
        original = json.load(f)

    # Compare T1 values: synthetic should be 0.7× original
    for qname in list(original["qubits"].keys())[:5]:
        orig_t1 = original["qubits"][qname]["t1_us"]
        synth_t1 = synth_json["qubits"][qname]["t1_us"]
        expected = round(orig_t1 * 0.7, 4)
        check(f"T1 scaling {qname}: {orig_t1:.2f} × 0.7 = {expected:.2f}",
              abs(synth_t1 - expected) < 0.01,
              f"got {synth_t1}")

    # T2 should not exceed 2*T1 after perturbation
    for qname, qdata in synth_json["qubits"].items():
        check(f"T2 ≤ 2*T1 for {qname}",
              qdata["t2_us"] <= 2.0 * qdata["t1_us"] + 0.001,
              f"T1={qdata['t1_us']}, T2={qdata['t2_us']}")
        break  # just check first one in detail, the rest by constraint

    # Verify ALL qubits satisfy T2 ≤ 2*T1
    violations = [
        qname for qname, qdata in synth_json["qubits"].items()
        if qdata["t2_us"] > 2.0 * qdata["t1_us"] + 0.001
    ]
    check("T2 ≤ 2*T1 constraint holds for all qubits",
          len(violations) == 0,
          f"violations: {violations[:3]}")

except Exception as e:
    check("E9.4 T1 scaling", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E9.5: All Perturbation Types ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.calibration_adapters.synthetic import SyntheticAdapter

    adapter = IQMv2Adapter()
    synth = SyntheticAdapter(adapter)
    base = adapter.load(cal_path)

    # scale_t2
    p1 = synth.perturb(base, {"scale_t2": 0.5})
    sample_q = list(p1.qubits.values())[0]
    orig_q = list(base.qubits.values())[0]
    check("scale_t2: T2 reduced",
          sample_q.t2_us < orig_q.t2_us)

    # scale_readout
    p2 = synth.perturb(base, {"scale_readout": 0.9})
    sample_q2 = list(p2.qubits.values())[0]
    check("scale_readout: readout reduced",
          sample_q2.readout_fidelity < orig_q.readout_fidelity)

    # scale_gate_fidelity
    p3 = synth.perturb(base, {"scale_gate_fidelity": 0.95})
    check("scale_gate_fidelity: gates present",
          len(p3.gates) > 0)
    sample_g = list(p3.gates.values())[0]
    orig_g = list(base.gates.values())[0]
    check("scale_gate_fidelity: fidelity reduced",
          sample_g.fidelity < orig_g.fidelity)

    # poison_qubit
    poison_target = list(base.qubits.keys())[10]  # pick QB-something
    p4 = synth.perturb(base, {"poison_qubit": poison_target})
    poisoned_q = p4.qubits[poison_target]
    healthy_q = base.qubits[poison_target]
    check(f"poison_qubit: {poison_target} T1 degraded",
          poisoned_q.t1_us < healthy_q.t1_us)
    check(f"poison_qubit: {poison_target} readout degraded",
          poisoned_q.readout_fidelity < healthy_q.readout_fidelity)

    # uniform_noise
    p5 = synth.perturb(base, {"uniform_t1": 25.0, "uniform_t2": 12.0})
    t1_vals = [q.t1_us for q in p5.qubits.values()]
    t2_vals = [q.t2_us for q in p5.qubits.values()]
    check("uniform_noise: all T1 = 25.0",
          all(abs(v - 25.0) < 0.001 for v in t1_vals),
          f"unique T1s: {set(round(v, 2) for v in t1_vals)}")
    check("uniform_noise: all T2 = 12.0",
          all(abs(v - 12.0) < 0.001 for v in t2_vals),
          f"unique T2s: {set(round(v, 2) for v in t2_vals)}")

    # improve_all
    p6 = synth.perturb(base, {"improve_all": 2.0})
    improved_q = list(p6.qubits.values())[0]
    check("improve_all: T1 doubled",
          abs(improved_q.t1_us - orig_q.t1_us * 2.0) < 0.01)
    check("improve_all: gate error halved",
          improved_q.single_gate_error < orig_q.single_gate_error)
    check("improve_all: readout improved",
          improved_q.readout_fidelity > orig_q.readout_fidelity)

    # Combined perturbation
    p7 = synth.perturb(base, {"scale_t1": 0.5, "scale_readout": 0.8})
    combined_q = list(p7.qubits.values())[0]
    check("Combined: T1 halved + readout degraded",
          combined_q.t1_us < orig_q.t1_us and
          combined_q.readout_fidelity < orig_q.readout_fidelity)

except Exception as e:
    check("E9.5 perturbation types", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E9.6: Key Validation ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Unknown key should raise ValueError
    caught = False
    try:
        synth.perturb(base, {"t1_factor": 0.7})  # wrong key name
    except ValueError as e:
        caught = True
        check("Key validation: error message mentions unrecognized",
              "Unrecognized" in str(e) or "unrecognized" in str(e).lower(),
              str(e))
    check("Key validation: ValueError raised for unknown key", caught)

    # Another wrong key
    caught2 = False
    try:
        synth.perturb(base, {"noise_level": 0.5})
    except ValueError:
        caught2 = True
    check("Key validation: rejects 'noise_level'", caught2)

except Exception as e:
    check("E9.6 key validation", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E9.7: Batch Generation ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    batch_dir = os.path.join(test_dir, "batch")
    perturbations = [
        {"scale_t1": f} for f in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    ]

    batch_results = generate_batch(
        cal_path, perturbations, batch_dir,
        name_template="t1_sweep_{index:03d}.json",
    )

    check("Batch: correct number of outputs",
          len(batch_results) == 6,
          f"got {len(batch_results)}")

    # Verify all files exist
    for r in batch_results:
        check(f"Batch file exists: {os.path.basename(r['output'])}",
              os.path.exists(r["output"]))

    # Verify T1 values are monotonically increasing across the batch
    t1_means = []
    for r in batch_results:
        with open(r["output"]) as f:
            data = json.load(f)
        mean_t1 = sum(q["t1_us"] for q in data["qubits"].values()) / len(data["qubits"])
        t1_means.append(mean_t1)

    check("Batch: T1 means are monotonically increasing",
          all(t1_means[i] < t1_means[i+1] for i in range(len(t1_means)-1)),
          f"means: {[round(m, 1) for m in t1_means]}")

except Exception as e:
    check("E9.7 batch generation", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E9.8: Twin Simulator Integration ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery
    from lumi_hpc_qc.sweep.noise_configs import NOISE_ENV_BY_NAME
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    # Build simple circuit + observable
    qc = QuantumCircuit(4)
    h_op = SparsePauliOp.from_list([("ZZZZ", 1.0)])

    # Load real and synthetic calibrations
    with open(cal_path) as f:
        real_cal_json = json.load(f)

    synth_path = os.path.join(test_dir, "synth_for_twin.json")
    generate_synthetic(cal_path, {"scale_t1": 0.5, "scale_readout": 0.8}, synth_path)
    with open(synth_path) as f:
        synth_cal_json = json.load(f)

    # Run twin battery on noise_full only for speed
    noise_full = [NOISE_ENV_BY_NAME["noise_full"]]

    real_battery = run_twin_battery(
        circuit=qc, observable=h_op,
        qubit_names=["QB6", "QB7", "QB13", "QB12"],
        calibration_data=real_cal_json,
        calibration_id="real",
        placement_id="test_p0",
        topology_hash="test_hash",
        environments=noise_full,
        seed=42, device="CPU",
    )

    synth_battery = run_twin_battery(
        circuit=qc, observable=h_op,
        qubit_names=["QB6", "QB7", "QB13", "QB12"],
        calibration_data=synth_cal_json,
        calibration_id="synthetic",
        placement_id="test_p0",
        topology_hash="test_hash",
        environments=noise_full,
        seed=42, device="CPU",
    )

    real_energy = real_battery.results[0].energy
    synth_energy = synth_battery.results[0].energy

    check("Twin sim: real calibration produces result",
          real_energy is not None)
    check("Twin sim: synthetic calibration produces result",
          synth_energy is not None)
    check("Twin sim: synthetic produces DIFFERENT energy than real",
          real_energy is not None and synth_energy is not None
          and abs(real_energy - synth_energy) > 0.001,
          f"real={real_energy}, synth={synth_energy}")

except Exception as e:
    check("E9.8 twin sim integration", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E9.9: Physical Constraint Edge Cases ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Extreme T1 degradation — T2 should be clamped to 2*T1
    extreme = synth.perturb(base, {"scale_t1": 0.1})
    for qname, qcal in extreme.qubits.items():
        if qcal.t2_us > 2.0 * qcal.t1_us + 0.001:
            check(f"Extreme: T2 ≤ 2*T1 for {qname}", False,
                  f"T1={qcal.t1_us}, T2={qcal.t2_us}")
            break
    else:
        check("Extreme T1 degradation: T2 ≤ 2*T1 for all qubits", True)

    # Readout clamping — should stay in [0.5, 1.0]
    low_ro = synth.perturb(base, {"scale_readout": 0.3})
    ro_vals = [q.readout_fidelity for q in low_ro.qubits.values()]
    check("Readout clamping: all ≥ 0.5",
          all(v >= 0.5 for v in ro_vals),
          f"min readout: {min(ro_vals)}")
    check("Readout clamping: all ≤ 1.0",
          all(v <= 1.0 for v in ro_vals))

    # Improve-all — readout should not exceed 1.0
    big_improve = synth.perturb(base, {"improve_all": 10.0})
    ro_improved = [q.readout_fidelity for q in big_improve.qubits.values()]
    check("Improve clamping: readout ≤ 1.0",
          all(v <= 1.0 for v in ro_improved))

except Exception as e:
    check("E9.9 edge cases", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"E9 VALIDATION RESULTS: {passed} passed, {failed} failed")
print(f"{'='*70}")

if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  - {e}")

if failed == 0:
    print("\nE9 VALIDATION: ALL CHECKS PASSED")
    sys.exit(0)
else:
    print(f"\nE9 VALIDATION: {failed} CHECKS FAILED")
    sys.exit(1)
