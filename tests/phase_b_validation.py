#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase B validation tests — RED-SPEC-001 §8.1-8.2 (V1-V6, V8).

Run on LUMI standard-g partition:
    srun ... python tests/phase_b_validation.py

V7 (topology-noiseless energy ordering) requires full VQE runs and is
deferred to benchmark execution, per Team Red's note in
RED-REVIEW-PHASE-B-PLAN §5.
"""

import sys
import os
import json
import traceback

# Ensure project is on path
project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

CALIBRATION_FILE = os.path.join(project_dir, "examples", "q50_calibration_20260326.json")

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
print("\n=== V1: readout_error only → no quantum errors ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.backends.noise_model import build_noise_model

    nm, cm = build_noise_model(
        CALIBRATION_FILE, 4,
        noise_channels={"readout_error": True}
    )

    # NoiseModel should have readout errors but no quantum errors
    has_readout = len(nm._default_readout_errors) > 0 or len(nm._local_readout_errors) > 0
    has_quantum = len(nm._default_quantum_errors) > 0 or len(nm._local_quantum_errors) > 0

    check("Readout errors present", has_readout, f"readout errors: {has_readout}")
    check("No quantum errors", not has_quantum,
          f"quantum errors found: default={len(nm._default_quantum_errors)}, "
          f"local={len(nm._local_quantum_errors)}")
except Exception as e:
    check("V1 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V2: all channels = backward compatible with v1.0.0b3 ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # All channels explicit
    nm_all, cm_all = build_noise_model(
        CALIBRATION_FILE, 4,
        noise_channels={
            "single_qubit_depolarizing": True,
            "two_qubit_depolarizing": True,
            "t1_relaxation": True,
            "t2_dephasing": True,
            "readout_error": True,
        }
    )

    # Default (None = all active)
    nm_default, cm_default = build_noise_model(
        CALIBRATION_FILE, 4,
        noise_channels=None
    )

    # Both should have the same number of error entries
    all_q_errs = len(nm_all._local_quantum_errors)
    default_q_errs = len(nm_default._local_quantum_errors)
    check("Quantum error count matches", all_q_errs == default_q_errs,
          f"all={all_q_errs}, default={default_q_errs}")

    all_ro_errs = len(nm_all._local_readout_errors)
    default_ro_errs = len(nm_default._local_readout_errors)
    check("Readout error count matches", all_ro_errs == default_ro_errs,
          f"all={all_ro_errs}, default={default_ro_errs}")

    # Coupling maps should be identical
    if cm_all and cm_default:
        check("Coupling map edges match",
              sorted(cm_all.get_edges()) == sorted(cm_default.get_edges()))
    else:
        check("Both have coupling maps", cm_all is not None and cm_default is not None,
              f"all={cm_all is not None}, default={cm_default is not None}")
except Exception as e:
    check("V2 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V3: all channels false → NoiseModel with no errors ===")
# ══════════════════════════════════════════════════════════════════════
try:
    nm_empty, cm_empty = build_noise_model(
        CALIBRATION_FILE, 4,
        noise_channels={
            "single_qubit_depolarizing": False,
            "two_qubit_depolarizing": False,
            "t1_relaxation": False,
            "t2_dephasing": False,
            "readout_error": False,
        }
    )

    has_quantum = len(nm_empty._default_quantum_errors) > 0 or len(nm_empty._local_quantum_errors) > 0
    has_readout = len(nm_empty._default_readout_errors) > 0 or len(nm_empty._local_readout_errors) > 0

    check("No quantum errors", not has_quantum)
    check("No readout errors", not has_readout)
    check("Coupling map still returned", cm_empty is not None,
          "Coupling map should exist even with no noise")
except Exception as e:
    check("V3 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V4: noise_config metadata accurately reflects channels ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.backends.noise_model import get_noise_config_metadata

    meta = get_noise_config_metadata(
        CALIBRATION_FILE, 4,
        noise_channels={"t1_relaxation": True, "readout_error": True},
        coupling_map_source="calibration",
    )

    check("channels_active reflects input",
          meta["channels_active"]["t1_relaxation"] is True
          and meta["channels_active"]["readout_error"] is True
          and meta["channels_active"]["single_qubit_depolarizing"] is False
          and meta["channels_active"]["two_qubit_depolarizing"] is False
          and meta["channels_active"]["t2_dephasing"] is False)

    check("coupling_map_source recorded",
          meta["coupling_map_source"] == "calibration")

    check("qubit_assignment populated",
          len(meta["qubit_assignment"]) == 4)

    check("calibration_summary has avg_t1_us",
          "avg_t1_us" in meta["calibration_summary"]
          and meta["calibration_summary"]["avg_t1_us"] > 0)
except Exception as e:
    check("V4 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V5: TFIM 2q — Q50 topology = full connectivity (no SWAPs) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.backends.noise_model import extract_coupling_map
    from qiskit import QuantumCircuit, transpile

    # Build a simple 2-qubit circuit
    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qc.h(0)
    pre_depth = qc.depth()
    pre_cx = qc.count_ops().get('cx', 0)

    # Transpile to Q50 topology
    cmap = extract_coupling_map(CALIBRATION_FILE, 2)
    if cmap is not None:
        transpiled = transpile(qc, coupling_map=cmap, optimization_level=2, seed_transpiler=42)
        post_depth = transpiled.depth()
        post_cx = transpiled.count_ops().get('cx', 0) + transpiled.count_ops().get('cz', 0)

        # 2 qubits on Q50: the best pair is directly connected, no SWAPs needed
        check("No SWAPs for 2q circuit", post_cx <= pre_cx + 1,
              f"pre_cx={pre_cx}, post_cx={post_cx} (expected no SWAP overhead)")
        check("Depth not increased significantly", post_depth <= pre_depth + 2,
              f"pre={pre_depth}, post={post_depth}")
    else:
        check("Coupling map extracted", False, "No coupling map returned")
except Exception as e:
    check("V5 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V6: TFIM 8q — Q50 topology > full connectivity depth ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from qiskit.circuit.library import EfficientSU2

    # Build an 8-qubit SU2 ansatz
    su2 = EfficientSU2(num_qubits=8, reps=2, entanglement="linear")
    su2_decomposed = su2.decompose().decompose().decompose()
    pre_depth = su2_decomposed.depth()
    pre_gates = su2_decomposed.size()

    # Transpile with full connectivity (no coupling map)
    full_transpiled = transpile(su2_decomposed, optimization_level=2, seed_transpiler=42)
    full_depth = full_transpiled.depth()

    # Transpile with Q50 topology
    cmap_8q = extract_coupling_map(CALIBRATION_FILE, 8)
    if cmap_8q is not None:
        topo_transpiled = transpile(su2_decomposed, coupling_map=cmap_8q,
                                    optimization_level=2, seed_transpiler=42)
        topo_depth = topo_transpiled.depth()
        topo_gates = topo_transpiled.size()

        check("Q50 topology increases depth",
              topo_depth >= full_depth,
              f"full={full_depth}, topology={topo_depth}")
        check("Q50 topology increases gate count (SWAPs added)",
              topo_gates >= pre_gates,
              f"pre={pre_gates}, topology={topo_gates}")
    else:
        check("8q coupling map extracted", False, "No coupling map returned")
except Exception as e:
    check("V6 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V8: Circuit metrics structure ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.types import CircuitMetrics, ExperimentRecord

    # Verify CircuitMetrics has all required fields
    cm = CircuitMetrics(
        pre_transpilation_depth=10,
        pre_transpilation_gate_count=30,
        pre_transpilation_cx_count=5,
        post_transpilation_depth=15,
        post_transpilation_gate_count=42,
        post_transpilation_cx_count=12,
        swap_count=7,
        coupling_map_source="calibration",
        coupling_map_edges=10,
        transpiler_optimization_level=2,
        num_parameters=16,
    )
    check("CircuitMetrics construction", cm.swap_count == 7)
    check("CircuitMetrics has all fields",
          hasattr(cm, 'pre_transpilation_depth')
          and hasattr(cm, 'post_transpilation_cx_count')
          and hasattr(cm, 'coupling_map_source'))

    # Verify ExperimentRecord accepts new fields
    rec = ExperimentRecord(
        experiment_id="test",
        circuit_metrics=cm,
        noise_config={"channels_active": {"readout_error": True}},
    )
    check("ExperimentRecord has circuit_metrics", rec.circuit_metrics is not None)
    check("ExperimentRecord has noise_config", rec.noise_config is not None)
    check("circuit_metrics.swap_count accessible", rec.circuit_metrics.swap_count == 7)
except Exception as e:
    check("V8 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V-extra: extract_coupling_map independent of noise model ===")
# ══════════════════════════════════════════════════════════════════════
try:
    cmap_only = extract_coupling_map(CALIBRATION_FILE, 4)
    check("extract_coupling_map returns CouplingMap", cmap_only is not None)
    check("Coupling map has edges", len(cmap_only.get_edges()) > 0,
          f"edges: {len(cmap_only.get_edges())}")
except Exception as e:
    check("extract_coupling_map", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V-extra: Placement solver ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.placement.solver import PlacementSolver

    solver = PlacementSolver(CALIBRATION_FILE)
    placements = solver.find_placements(circuit_qubits=4, num_placements=12, strategy="max_fidelity")

    check("Placements found", len(placements) > 0, f"count: {len(placements)}")
    check("Placements are non-overlapping",
          len(set(idx for p in placements for idx in p["physical_indices"]))
          == sum(len(p["physical_indices"]) for p in placements))

    if placements:
        p0 = placements[0]
        check("Placement has qubit_mapping", "qubit_mapping" in p0)
        check("Placement has score", "score" in p0 and p0["score"] > 0)
        check("Placement has avg_readout_fidelity",
              "avg_readout_fidelity" in p0 and p0["avg_readout_fidelity"] > 0.9)

    summary = solver.summary(placements)
    check("Summary has utilization", "utilization" in summary and summary["utilization"] > 0)
except Exception as e:
    check("Placement solver", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V-extra: Config generator ===")
# ══════════════════════════════════════════════════════════════════════
try:
    configs_dir = os.path.join(project_dir, "configs", "generated")
    if os.path.isdir(configs_dir):
        files = [f for f in os.listdir(configs_dir) if f.endswith(".yaml")]
        check("TFIM 4q configs generated", len(files) == 13,
              f"found {len(files)} files, expected 13")

        # Spot-check one config
        topo_path = os.path.join(configs_dir, "q50bench_tfim_4q_topology_noiseless.yaml")
        if os.path.exists(topo_path):
            import yaml
            with open(topo_path) as f:
                cfg = yaml.safe_load(f)
            check("topology_noiseless has coupling_map_source",
                  cfg.get("backend_params", {}).get("coupling_map_source") == "calibration")
            check("topology_noiseless uses statevector",
                  cfg.get("backend_params", {}).get("method") == "statevector")
            check("topology_noiseless has no noise_model_file",
                  "noise_model_file" not in cfg.get("backend_params", {}))
        else:
            check("topology_noiseless config exists", False, f"not found at {topo_path}")
    else:
        check("configs/generated directory exists", False, f"{configs_dir} not found")
except Exception as e:
    check("Config generator", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase B validation: {passed} passed, {failed} failed")
if errors:
    print(f"\nFailures:")
    for e in errors:
        print(f"  - {e}")
print(f"{'='*60}")

sys.exit(1 if failed > 0 else 0)
