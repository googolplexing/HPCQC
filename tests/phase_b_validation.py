#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase B validation tests — RED-SPEC-001 §8.1-8.2 (V1-V6, V8).

V7 (topology-noiseless energy ordering) requires full VQE runs and is
deferred to benchmark execution.
"""

import sys
import os
import json
import traceback

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

CALIBRATION_FILE = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")

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


def noise_model_error_counts(nm):
    """Extract error counts from NoiseModel using the public to_dict() API."""
    d = nm.to_dict()
    quantum_errors = [e for e in d.get("errors", []) if e.get("type") == "qerror"]
    readout_errors = [e for e in d.get("errors", []) if e.get("type") == "roerror"]
    return len(quantum_errors), len(readout_errors)


# ══════════════════════════════════════════════════════════════════════
print("\n=== V1: readout_error only → no quantum errors ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.backends.noise_model import build_noise_model

    nm, cm = build_noise_model(
        CALIBRATION_FILE, 4,
        noise_channels={"readout_error": True}
    )

    n_quantum, n_readout = noise_model_error_counts(nm)
    check("Readout errors present", n_readout > 0, f"readout errors: {n_readout}")
    check("No quantum errors", n_quantum == 0,
          f"quantum errors found: {n_quantum}")
except Exception as e:
    check("V1 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V2: all channels = backward compatible with v1.0.0b3 ===")
# ══════════════════════════════════════════════════════════════════════
try:
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
    nm_default, cm_default = build_noise_model(
        CALIBRATION_FILE, 4,
        noise_channels=None
    )

    all_q, all_ro = noise_model_error_counts(nm_all)
    def_q, def_ro = noise_model_error_counts(nm_default)
    check("Quantum error count matches", all_q == def_q,
          f"all={all_q}, default={def_q}")
    check("Readout error count matches", all_ro == def_ro,
          f"all={all_ro}, default={def_ro}")

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

    n_quantum, n_readout = noise_model_error_counts(nm_empty)
    check("No quantum errors", n_quantum == 0, f"found {n_quantum}")
    check("No readout errors", n_readout == 0, f"found {n_readout}")
    check("Coupling map still returned", cm_empty is not None)
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
print("\n=== V5: 2q circuit on Q50 topology — no SWAPs needed ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.backends.noise_model import extract_coupling_map
    from qiskit import QuantumCircuit, transpile

    # 2 qubits on full Q50 — top 2 by fidelity should be directly connected
    cmap_2q = extract_coupling_map(CALIBRATION_FILE, 2)
    if cmap_2q is not None and len(cmap_2q.get_edges()) > 0:
        qc = QuantumCircuit(2)
        qc.cx(0, 1)
        qc.h(0)

        transpiled = transpile(qc, coupling_map=cmap_2q, optimization_level=2, seed_transpiler=42)
        swap_count = transpiled.count_ops().get('swap', 0)

        check("2q circuit transpiles on Q50 topology", True)
        check("No SWAPs for directly-connected pair", swap_count == 0,
              f"SWAPs={swap_count}")
    else:
        # Top-2 qubits may not be directly connected — use 4 qubits instead
        cmap_4q = extract_coupling_map(CALIBRATION_FILE, 4)
        if cmap_4q and len(cmap_4q.get_edges()) > 0:
            qc = QuantumCircuit(4)
            qc.cx(0, 1)
            qc.h(0)
            transpiled = transpile(qc, coupling_map=cmap_4q, optimization_level=2, seed_transpiler=42)
            check("Small circuit transpiles on Q50 4q topology", True)
        else:
            check("Q50 coupling map has edges", False)
except Exception as e:
    check("V5 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V6: 8q circuit on Q50 topology — SWAPs needed ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from qiskit.circuit.library import EfficientSU2

    # SU2 with "full" entanglement on 8 qubits — all-to-all CX gates
    # On Q50's heavy-hex topology, non-adjacent pairs need SWAPs
    su2_8q = EfficientSU2(num_qubits=8, reps=2, entanglement="full")
    su2_dec = su2_8q.decompose().decompose().decompose()

    # Full connectivity (no routing)
    full_transpiled = transpile(su2_dec, optimization_level=2, seed_transpiler=42)
    full_depth = full_transpiled.depth()
    full_gates = full_transpiled.size()

    # Q50 topology (routing needed)
    cmap_8q = extract_coupling_map(CALIBRATION_FILE, 8)
    if cmap_8q and len(cmap_8q.get_edges()) >= 6:
        topo_transpiled = transpile(su2_dec, coupling_map=cmap_8q,
                                    optimization_level=2, seed_transpiler=42)
        topo_depth = topo_transpiled.depth()
        topo_gates = topo_transpiled.size()

        check("Q50 topology increases depth vs full connectivity",
              topo_depth >= full_depth,
              f"full={full_depth}, Q50={topo_depth}")
        check("Q50 topology adds gates (routing overhead)",
              topo_gates > full_gates,
              f"full={full_gates}, Q50={topo_gates}")
        print(f"    Full connectivity: depth={full_depth}, gates={full_gates}")
        print(f"    Q50 topology:      depth={topo_depth}, gates={topo_gates}")
        print(f"    Routing overhead:  +{topo_depth - full_depth} depth, +{topo_gates - full_gates} gates")
    else:
        check("8q Q50 coupling map sufficient", False,
              f"edges: {len(cmap_8q.get_edges()) if cmap_8q else 0}")
except Exception as e:
    check("V6 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== V8: Circuit metrics structure ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.types import CircuitMetrics, ExperimentRecord

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
            check("topology_noiseless config exists", False)
    else:
        check("configs/generated directory exists", False)
except Exception as e:
    check("Config generator", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# Individual channel isolation tests
# ══════════════════════════════════════════════════════════════════════
print("\n=== V-extra: Individual channel isolation ===")
try:
    # Only 1q depolarizing
    nm_1q, _ = build_noise_model(CALIBRATION_FILE, 4,
                                  noise_channels={"single_qubit_depolarizing": True})
    nq_1q, nro_1q = noise_model_error_counts(nm_1q)
    check("1q_only: has quantum errors", nq_1q > 0, f"count={nq_1q}")
    check("1q_only: no readout errors", nro_1q == 0, f"count={nro_1q}")

    # Only T1
    nm_t1, _ = build_noise_model(CALIBRATION_FILE, 4,
                                  noise_channels={"t1_relaxation": True})
    nq_t1, nro_t1 = noise_model_error_counts(nm_t1)
    check("t1_only: has quantum errors (thermal)", nq_t1 > 0, f"count={nq_t1}")
    check("t1_only: no readout errors", nro_t1 == 0, f"count={nro_t1}")
except Exception as e:
    check("Channel isolation", False, f"Exception: {e}")
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
