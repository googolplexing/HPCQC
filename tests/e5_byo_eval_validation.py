#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E — E5: BYO Circuit Ingestion + Evaluation-Only Mode.

Tests circuit loading (QPY, QASM, script), connectivity extraction,
eval-only execution, and parameterization detection.

VE15: Non-parameterized circuit runs in evaluation-only mode (no optimizer).

Run on LUMI standard partition (CPU only):
    srun ... python tests/e5_byo_eval_validation.py

Expected: E5 VALIDATION: ALL CHECKS PASSED

RED-SPEC-002 §7 (BYO Circuit Ingestion)
"""

import sys
import os
import time
import traceback
import tempfile

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

import numpy as np

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


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.1: Circuit Loader — QASM String ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.circuit_loader import (
        load_circuit, load_from_config, extract_connectivity, LoadedCircuit,
    )
    from qiskit import QuantumCircuit

    # 3-qubit GHZ circuit as QASM 2.0 string
    ghz_qasm = """
OPENQASM 2.0;
include "qelib1.inc";
qreg q[3];
creg c[3];
h q[0];
cx q[0],q[1];
cx q[1],q[2];
measure q -> c;
"""
    loaded = load_circuit(qasm_string=ghz_qasm)
    check("QASM string: loads successfully", loaded.circuit is not None)
    check("QASM string: 3 qubits", loaded.num_qubits == 3,
          f"got {loaded.num_qubits}")
    check("QASM string: not parameterized", not loaded.is_parameterized)
    check("QASM string: 0 parameters", loaded.num_parameters == 0)
    check("QASM string: connectivity extracted",
          len(loaded.connectivity) >= 2,
          f"got {loaded.connectivity}")
    check("QASM string: connectivity has (0,1) and (1,2)",
          (0, 1) in loaded.connectivity and (1, 2) in loaded.connectivity,
          f"got {loaded.connectivity}")
    check("QASM string: source recorded",
          "qasm" in loaded.source.lower(),
          f"source={loaded.source}")

except Exception as e:
    check("E5.1 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.2: Circuit Loader — QPY File ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Create a 4-qubit star circuit and save as QPY
    tmpdir = tempfile.mkdtemp(prefix="e5_test_")

    star_qc = QuantumCircuit(4)
    star_qc.h(0)
    star_qc.cx(0, 1)
    star_qc.cx(0, 2)
    star_qc.cx(0, 3)
    star_qc.measure_all()

    qpy_path = os.path.join(tmpdir, "star_4q.qpy")
    from qiskit.qpy import dump as qpy_dump
    with open(qpy_path, "wb") as f:
        qpy_dump([star_qc], f)

    loaded_qpy = load_circuit(qpy_file=qpy_path)
    check("QPY file: loads successfully", loaded_qpy.circuit is not None)
    check("QPY file: 4 qubits", loaded_qpy.num_qubits == 4,
          f"got {loaded_qpy.num_qubits}")
    check("QPY file: not parameterized", not loaded_qpy.is_parameterized)
    check("QPY file: star connectivity = 3 pairs",
          len(loaded_qpy.connectivity) == 3,
          f"got {loaded_qpy.connectivity}")
    check("QPY file: hub qubit 0 in all pairs",
          all(0 in pair for pair in loaded_qpy.connectivity),
          f"connectivity={loaded_qpy.connectivity}")
    check("QPY file: source mentions qpy",
          "qpy" in loaded_qpy.source.lower(),
          f"source={loaded_qpy.source}")

except Exception as e:
    check("E5.2 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.3: Circuit Loader — Python Script ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Write a script that builds a circuit
    script_path = os.path.join(tmpdir, "build_bell.py")
    with open(script_path, "w") as f:
        f.write("""
from qiskit import QuantumCircuit

def build_circuit():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc
""")

    loaded_script = load_circuit(
        script_file=script_path,
        script_function="build_circuit",
    )
    check("Script: loads successfully", loaded_script.circuit is not None)
    check("Script: 2 qubits", loaded_script.num_qubits == 2,
          f"got {loaded_script.num_qubits}")
    check("Script: not parameterized", not loaded_script.is_parameterized)
    check("Script: connectivity = [(0,1)]",
          loaded_script.connectivity == [(0, 1)],
          f"got {loaded_script.connectivity}")
    check("Script: source mentions script",
          "script" in loaded_script.source.lower(),
          f"source={loaded_script.source}")

except Exception as e:
    check("E5.3 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.4: Circuit Loader — Config Interface ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Test load_from_config with QPY file
    config_qpy = {"circuit_file": qpy_path}
    loaded_cfg = load_from_config(config_qpy)
    check("Config QPY: loads from config dict", loaded_cfg.num_qubits == 4)

    # Test load_from_config with script
    config_script = {
        "circuit_script": script_path,
        "circuit_function": "build_circuit",
    }
    loaded_cfg2 = load_from_config(config_script)
    check("Config script: loads from config dict", loaded_cfg2.num_qubits == 2)

    # Test load_from_config with inline QASM
    config_qasm = {"circuit_qasm": ghz_qasm}
    loaded_cfg3 = load_from_config(config_qasm)
    check("Config QASM: loads from config dict", loaded_cfg3.num_qubits == 3)

except Exception as e:
    check("E5.4 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.5: Parameterized Circuit Detection ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from qiskit.circuit import Parameter

    # Build a parameterized circuit
    theta = Parameter("θ")
    param_qc = QuantumCircuit(2)
    param_qc.ry(theta, 0)
    param_qc.cx(0, 1)

    # Save as QPY and load
    param_path = os.path.join(tmpdir, "param_circuit.qpy")
    with open(param_path, "wb") as f:
        qpy_dump([param_qc], f)

    loaded_param = load_circuit(qpy_file=param_path)
    check("Parameterized: detected as parameterized",
          loaded_param.is_parameterized,
          f"is_parameterized={loaded_param.is_parameterized}")
    check("Parameterized: num_parameters = 1",
          loaded_param.num_parameters == 1,
          f"got {loaded_param.num_parameters}")

    # Verify eval-only runner rejects parameterized circuits
    from lumi_hpc_qc.sweep.eval_runner import evaluate_circuit
    rejected = False
    try:
        evaluate_circuit(loaded_param)
    except ValueError as e:
        if "parameterized" in str(e).lower():
            rejected = True
    check("VE15 (partial): eval-only rejects parameterized circuit", rejected)

except Exception as e:
    check("E5.5 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.6: VE15 — Eval-Only Mode (exact, no observable) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.eval_runner import evaluate_circuit, EvalResult

    # Build a simple Bell circuit (no measurements for density_matrix)
    bell = QuantumCircuit(2)
    bell.h(0)
    bell.cx(0, 1)

    loaded_bell = LoadedCircuit(
        circuit=bell,
        num_qubits=2,
        num_parameters=0,
        is_parameterized=False,
        connectivity=[(0, 1)],
        source="test:bell",
    )

    result = evaluate_circuit(
        loaded_bell,
        method="density_matrix",
        shots=0,
        seed=42,
        device="CPU",
    )
    check("VE15: eval-only completes without error",
          result.error is None,
          f"error={result.error}")
    check("VE15: execution_time > 0",
          result.execution_time_s > 0)
    check("VE15: no energy (no observable provided)",
          result.energy is None)
    print(f"    ({result.execution_time_s:.3f}s, backend={result.backend_used})")

except Exception as e:
    check("E5.6 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.7: VE15 — Eval-Only Mode (exact, with observable) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from qiskit.quantum_info import SparsePauliOp

    # Observable: ZZ (measures correlation between qubits)
    zz_obs = SparsePauliOp.from_list([("ZZ", 1.0)])

    result_obs = evaluate_circuit(
        loaded_bell,
        observable=zz_obs,
        method="density_matrix",
        shots=0,
        seed=42,
        device="CPU",
    )
    check("VE15 observable: completes without error",
          result_obs.error is None,
          f"error={result_obs.error}")
    check("VE15 observable: energy computed",
          result_obs.energy is not None)
    # Bell state |00⟩+|11⟩: ⟨ZZ⟩ = 1.0 (perfect correlation)
    if result_obs.energy is not None:
        check("VE15 observable: Bell ⟨ZZ⟩ = 1.0",
              abs(result_obs.energy - 1.0) < 1e-6,
              f"got {result_obs.energy:.6f}")
    print(f"    energy={result_obs.energy}, time={result_obs.execution_time_s:.3f}s")

except Exception as e:
    check("E5.7 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.8: VE15 — Eval-Only GHZ 3q (reference circuit) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # GHZ-3: reference characterization circuit
    ghz3 = QuantumCircuit(3)
    ghz3.h(0)
    ghz3.cx(0, 1)
    ghz3.cx(1, 2)

    loaded_ghz3 = LoadedCircuit(
        circuit=ghz3,
        num_qubits=3,
        num_parameters=0,
        is_parameterized=False,
        connectivity=[(0, 1), (1, 2)],
        source="test:ghz3",
    )

    # Observable: ZZZ (all-qubit correlation)
    zzz_obs = SparsePauliOp.from_list([("ZZZ", 1.0)])

    result_ghz = evaluate_circuit(
        loaded_ghz3,
        observable=zzz_obs,
        method="density_matrix",
        shots=0,
        seed=42,
        device="CPU",
    )
    check("GHZ-3: completes without error",
          result_ghz.error is None,
          f"error={result_ghz.error}")
    # GHZ state |000⟩+|111⟩: ⟨ZZZ⟩ = 0 (odd-qubit parity cancellation)
    if result_ghz.energy is not None:
        check("GHZ-3: ⟨ZZZ⟩ = 0 (odd-qubit parity)",
              abs(result_ghz.energy) < 1e-6,
              f"got {result_ghz.energy:.6f}")

    # Same circuit with different observable: ⟨ZZI⟩ = 1.0 for GHZ
    zzi_obs = SparsePauliOp.from_list([("ZZI", 1.0)])
    result_zzi = evaluate_circuit(
        loaded_ghz3,
        observable=zzi_obs,
        method="density_matrix",
        shots=0,
        seed=42,
        device="CPU",
    )
    if result_zzi.energy is not None:
        check("GHZ-3: ⟨ZZI⟩ = 1.0",
              abs(result_zzi.energy - 1.0) < 1e-6,
              f"got {result_zzi.energy:.6f}")

    # ⟨XII⟩ should be 0 for GHZ (no net magnetization)
    x_obs = SparsePauliOp.from_list([("XII", 1.0)])
    result_x = evaluate_circuit(
        loaded_ghz3,
        observable=x_obs,
        method="density_matrix",
        shots=0,
        seed=42,
        device="CPU",
    )
    if result_x.energy is not None:
        check("GHZ-3: ⟨XII⟩ = 0 (no single-qubit magnetization)",
              abs(result_x.energy) < 1e-6,
              f"got {result_x.energy:.6f}")

except Exception as e:
    check("E5.8 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.9: Star 4q Reference Circuit (Topology Library) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Build star-topology characterization circuit from topology library
    from lumi_hpc_qc.sweep.topology_library import TOPOLOGY_LIBRARY

    star_spec = TOPOLOGY_LIBRARY["4q_star"]
    check("Topology library: 4q_star exists",
          star_spec is not None)
    check("Topology library: 4q_star has 4 qubits",
          star_spec["qubits"] == 4)
    check("Topology library: 4q_star has 3 edges",
          len(star_spec["edges"]) == 3)

    # Build a reference GHZ-like circuit for star topology
    star_ref = QuantumCircuit(4)
    star_ref.h(0)  # hub
    for _, target in star_spec["edges"]:
        star_ref.cx(0, target)

    loaded_star_ref = LoadedCircuit(
        circuit=star_ref,
        num_qubits=4,
        num_parameters=0,
        is_parameterized=False,
        connectivity=star_spec["edges"],
        source="topology_library:4q_star",
    )

    # ⟨ZZZZ⟩ = 1.0 for GHZ-like state
    zzzz_obs = SparsePauliOp.from_list([("ZZZZ", 1.0)])
    result_star = evaluate_circuit(
        loaded_star_ref,
        observable=zzzz_obs,
        method="density_matrix",
        shots=0,
        seed=42,
        device="CPU",
    )
    check("Star 4q: eval-only completes",
          result_star.error is None,
          f"error={result_star.error}")
    if result_star.energy is not None:
        check("Star 4q: ⟨ZZZZ⟩ = 1.0 (GHZ-like)",
              abs(result_star.energy - 1.0) < 1e-6,
              f"got {result_star.energy:.6f}")

except Exception as e:
    check("E5.9 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.10: Reproducibility (same seed = same energy) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Run the same evaluation twice with the same seed
    r1 = evaluate_circuit(loaded_bell, observable=zz_obs,
                          method="density_matrix", shots=0, seed=42, device="CPU")
    r2 = evaluate_circuit(loaded_bell, observable=zz_obs,
                          method="density_matrix", shots=0, seed=42, device="CPU")
    check("Reproducibility: same seed = same energy",
          r1.energy is not None and r2.energy is not None
          and abs(r1.energy - r2.energy) < 1e-10,
          f"r1={r1.energy}, r2={r2.energy}")

except Exception as e:
    check("E5.10 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.11: Error Handling ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # No source provided
    err_caught = False
    try:
        load_circuit()
    except ValueError:
        err_caught = True
    check("Error: no source → ValueError", err_caught)

    # Multiple sources provided
    err_caught = False
    try:
        load_circuit(qpy_file="a.qpy", qasm_string="OPENQASM 2.0;")
    except ValueError:
        err_caught = True
    check("Error: multiple sources → ValueError", err_caught)

    # Non-existent file
    err_caught = False
    try:
        load_circuit(qpy_file="/nonexistent/path.qpy")
    except FileNotFoundError:
        err_caught = True
    check("Error: missing file → FileNotFoundError", err_caught)

except Exception as e:
    check("E5.11 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# CLEANUP
# ══════════════════════════════════════════════════════════════════════
try:
    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)
except Exception:
    pass


# ══════════════════════════════════════════════════════════════════════
print("\n=== E5.8: Cross-Path Validation — Shot-Based vs Exact (F1 Fix) ===")
# ══════════════════════════════════════════════════════════════════════
# RED-FINDING-EVAL-RUNNER-v1.0: Every energy computation path must be
# validated against the exact path for the same circuit.
try:
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp
    from lumi_hpc_qc.sweep.circuit_loader import LoadedCircuit
    from lumi_hpc_qc.sweep.eval_runner import evaluate_circuit, _energy_from_counts

    # ── TFIM 4q Hamiltonian: H = -ZZ₁₂ - ZZ₂₃ - ZZ₃₄ - X₁ - X₂ - X₃ - X₄ ──
    tfim_terms = [
        ("IIZZ", -1.0), ("IZZI", -1.0), ("ZZII", -1.0),
        ("IIIX", -1.0), ("IIXI", -1.0), ("IXII", -1.0), ("XIII", -1.0),
    ]
    tfim_labels = [t[0] for t in tfim_terms]
    tfim_coeffs = [t[1] for t in tfim_terms]
    tfim_obs = SparsePauliOp(tfim_labels, tfim_coeffs)

    # Simple circuit: identity (produces |0000⟩)
    qc_id = QuantumCircuit(4)
    loaded_id = LoadedCircuit(
        circuit=qc_id,
        source="inline",
        num_qubits=4,
        num_parameters=0,
        is_parameterized=False,
        connectivity=[],
    )

    # ── Exact evaluation (shots=0, statevector) ──
    exact_result = evaluate_circuit(
        loaded_id, observable=tfim_obs,
        method="statevector", shots=0, seed=42,
    )
    check("E5.8a: Exact evaluation succeeds",
          exact_result.error is None,
          f"Error: {exact_result.error}")
    check("E5.8b: Exact energy for |0000⟩ is -3.0",
          exact_result.energy is not None and abs(exact_result.energy - (-3.0)) < 1e-10,
          f"Got {exact_result.energy}, expected -3.0")

    # ── Shot-based evaluation (shots=4096, density_matrix) ──
    shot_result = evaluate_circuit(
        loaded_id, observable=tfim_obs,
        method="density_matrix", shots=4096, seed=42,
    )
    check("E5.8c: Shot-based evaluation succeeds",
          shot_result.error is None,
          f"Error: {shot_result.error}")

    # |0000⟩ is deterministic: X terms contribute exactly 0.0, ZZ terms
    # contribute exactly -1.0 each. With proper basis rotation, shot noise
    # only affects X-group circuits. Tolerance: 0.3 (generous for 4096 shots).
    if shot_result.energy is not None and exact_result.energy is not None:
        delta = abs(shot_result.energy - exact_result.energy)
        check("E5.8d: Shot-based energy matches exact within tolerance",
              delta < 0.3,
              f"|{shot_result.energy:.4f} - {exact_result.energy:.4f}| = {delta:.4f} > 0.3")
        # The OLD bug produced -7.0 for |0000⟩. Verify we're NOT getting that.
        check("E5.8e: Shot-based energy is NOT the buggy -7.0",
              abs(shot_result.energy - (-7.0)) > 1.0,
              f"Got {shot_result.energy:.4f} — still hitting the F1 bug!")
    else:
        check("E5.8d: Shot-based energy matches exact within tolerance",
              False, "Energy is None")
        check("E5.8e: Shot-based energy is NOT the buggy -7.0",
              False, "Energy is None")

    # ── Deprecation guard: _energy_from_counts must reject X/Y terms ──
    dummy_counts = {"0000": 1024, "0001": 1024, "0010": 1024, "0011": 1024}
    try:
        _energy_from_counts(dummy_counts, tfim_obs, 4)
        check("E5.8f: _energy_from_counts rejects TFIM (has X terms)",
              False, "Did not raise ValueError")
    except ValueError as ve:
        check("E5.8f: _energy_from_counts rejects TFIM (has X terms)",
              "X" in str(ve),
              f"Wrong error: {ve}")

    # ── Pure-Z observable: _energy_from_counts should still work ──
    zz_obs = SparsePauliOp(["IIZZ", "IZZI", "ZZII"], [-1.0, -1.0, -1.0])
    try:
        zz_energy = _energy_from_counts({"0000": 4096}, zz_obs, 4)
        check("E5.8g: _energy_from_counts accepts pure-Z observable",
              abs(zz_energy - (-3.0)) < 1e-10,
              f"Got {zz_energy}, expected -3.0")
    except ValueError as ve:
        check("E5.8g: _energy_from_counts accepts pure-Z observable",
              False, f"Unexpected rejection: {ve}")

    # ── Cross-check with a non-trivial state ──
    # Bell-like state: H on q0, CX q0→q1 → (|00⟩+|11⟩)/√2 on first 2 qubits
    qc_bell = QuantumCircuit(4)
    qc_bell.h(0)
    qc_bell.cx(0, 1)
    loaded_bell = LoadedCircuit(
        circuit=qc_bell, source="inline", num_qubits=4,
        num_parameters=0, is_parameterized=False,
        connectivity=[(0, 1)],
    )
    exact_bell = evaluate_circuit(
        loaded_bell, observable=tfim_obs,
        method="statevector", shots=0, seed=42,
    )
    shot_bell = evaluate_circuit(
        loaded_bell, observable=tfim_obs,
        method="density_matrix", shots=8192, seed=42,
    )
    if shot_bell.energy is not None and exact_bell.energy is not None:
        delta_bell = abs(shot_bell.energy - exact_bell.energy)
        check("E5.8h: Bell state cross-path within tolerance",
              delta_bell < 0.3,
              f"|{shot_bell.energy:.4f} - {exact_bell.energy:.4f}| = {delta_bell:.4f}")
    else:
        check("E5.8h: Bell state cross-path within tolerance",
              False, f"Energy None: exact={exact_bell.energy}, shot={shot_bell.energy}")

except Exception as e:
    traceback.print_exc()
    check("E5.8: Cross-path validation block", False, str(e))


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"E5 VALIDATION: {passed} passed, {failed} failed")
if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  ✗ {e}")
    print(f"\nE5 VALIDATION: FAILED ({failed} failures)")
    sys.exit(1)
else:
    print(f"\nE5 VALIDATION: ALL {passed} CHECKS PASSED")
    sys.exit(0)
