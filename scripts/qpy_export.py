# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""QPY export — serialise the optimised ansatz circuit from a completed VQE run.

V20 validation: QPY file is loadable by qpy.load() and round-trips correctly —
num_qubits, depth, and num_parameters (0, fully bound) all match.

Usage:
    python3 scripts/qpy_export.py <result_json> [--output <path.qpy>]

Example:
    python3 scripts/qpy_export.py \
        results/byo/302cc5e01c5d_17118012_result.json \
        --output results/byo/302cc5e01c5d_17118012_ansatz.qpy
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def export_qpy(result_json_path: str, output_path: str | None = None) -> str:
    """Rebuild optimised ansatz circuit from result JSON and serialise to QPY.

    Parameter binding mirrors workflow.py line 312:
        param_dict = dict(zip(ansatz.parameters, params))
        bound = circuit.assign_parameters(param_dict)

    Args:
        result_json_path: Path to *_result.json from a completed VQE run.
        output_path: Output .qpy path. Defaults to same dir as result JSON.

    Returns:
        Path to the written QPY file.
    """
    import numpy as np

    # ── Load result JSON ──────────────────────────────────────────────
    with open(result_json_path) as f:
        result = json.load(f)

    exp_id      = result["experiment_id"]
    config_dict = result["config"]
    convergence = result.get("convergence", {})
    iterations  = result.get("iterations", [])

    num_qubits  = config_dict["num_qubits"]
    ansatz_name = config_dict["ansatz"]

    print(f"Experiment:  {exp_id}")
    print(f"Ansatz:      {ansatz_name}, {num_qubits} qubits")
    print(f"Best energy: {convergence.get('best_energy')}")
    print(f"Error:       {convergence.get('relative_error_pct', '?')}%")
    print()

    # ── Extract best parameters ───────────────────────────────────────
    # Iterate in reverse to find the last iteration marked is_best=True
    best_params = None
    for it in reversed(iterations):
        if it.get("is_best"):
            best_params = it["parameters"]
            break
    if best_params is None and iterations:
        # Fallback: use final iteration
        best_params = iterations[-1]["parameters"]
    if best_params is None:
        raise ValueError("No parameter data found in result JSON")

    best_params = np.array(best_params, dtype=float)
    print(f"Parameters: {len(best_params)} values extracted from best iteration")

    # ── Reconstruct ExperimentConfig ─────────────────────────────────
    # Only fields used by the ansatz builder are needed.
    # ExperimentConfig is a dataclass with defaults — only set what we have.
    from lumi_hpc_qc.types import ExperimentConfig

    exp_config = ExperimentConfig(
        model            = config_dict.get("model", ""),
        model_params     = config_dict.get("model_params", {}),
        ansatz           = ansatz_name,
        ansatz_params    = config_dict.get("ansatz_params", {}),
        optimizer        = config_dict.get("optimizer", "l_bfgs_b"),
        optimizer_params = config_dict.get("optimizer_params", {}),
        gradient         = config_dict.get("gradient", "parameter_shift"),
        gradient_params  = config_dict.get("gradient_params", {}),
        initializer      = config_dict.get("initializer", "random"),
        initializer_params = config_dict.get("initializer_params", {}),
        backend          = config_dict.get("backend", "aer_gpu"),
        backend_params   = config_dict.get("backend_params", {}),
        precision        = config_dict.get("precision", "double"),
        num_qubits       = num_qubits,
        mode             = config_dict.get("mode", "interactive"),
    )

    # ── Discover plugins and build ansatz ─────────────────────────────
    # build() returns (QuantumCircuit, AnsatzMetadata)
    from lumi_hpc_qc.plugins.registry import PluginRegistry

    registry = PluginRegistry()
    registry.discover()

    ansatz_builder = registry.get_ansatz(ansatz_name)
    ansatz_circuit, ansatz_meta = ansatz_builder.build(num_qubits, exp_config)

    print(f"Ansatz built: depth={ansatz_circuit.depth()}, "
          f"num_parameters={ansatz_circuit.num_parameters}")

    # Verify parameter count matches stored result
    if ansatz_circuit.num_parameters != len(best_params):
        raise ValueError(
            f"Parameter count mismatch: circuit has "
            f"{ansatz_circuit.num_parameters} parameters but result JSON has "
            f"{len(best_params)} — ansatz config may not match stored result"
        )

    # ── Bind parameters using same approach as workflow.py ────────────
    # workflow.py line 312: param_dict = dict(zip(ansatz.parameters, params))
    # ansatz.parameters is a sorted ParameterView matching storage order
    param_dict = dict(zip(ansatz_circuit.parameters, best_params))
    bound_circuit = ansatz_circuit.assign_parameters(param_dict)

    print(f"Bound circuit: depth={bound_circuit.depth()}, "
          f"free_parameters={bound_circuit.num_parameters} (should be 0)")

    if bound_circuit.num_parameters != 0:
        raise ValueError(
            f"Binding failed: {bound_circuit.num_parameters} parameters remain unbound"
        )

    # ── Decompose to Aer primitive gates (if needed) ──────────────────
    # SU2 plugin already decomposes in build(), but run decompose_for_aer
    # anyway to guarantee QPY contains only primitive gates.
    # decompose_for_aer returns (circuit, num_rounds)
    from lumi_hpc_qc.backends.aer_gpu import decompose_for_aer

    decomposed, rounds = decompose_for_aer(bound_circuit)
    print(f"Decomposed: depth={decomposed.depth()}, "
          f"rounds needed={rounds}")
    print(f"Gate counts: {dict(decomposed.count_ops())}")

    # ── Serialise to QPY ──────────────────────────────────────────────
    # qpy.dump(programs, file_obj) — programs can be a single circuit or list
    from qiskit.qpy import dump as qpy_dump

    if output_path is None:
        result_dir  = os.path.dirname(os.path.abspath(result_json_path))
        output_path = os.path.join(result_dir, f"{exp_id}_ansatz.qpy")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "wb") as f:
        qpy_dump(decomposed, f)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"\nQPY written: {output_path} ({size_kb:.1f} KB)")

    # ── V20 verification: load back and check round-trip ─────────────
    # qpy.load(file_obj) returns list[QuantumCircuit]
    from qiskit.qpy import load as qpy_load

    with open(output_path, "rb") as f:
        loaded_circuits = qpy_load(f)

    loaded = loaded_circuits[0]

    errors = []
    if loaded.num_qubits != num_qubits:
        errors.append(
            f"num_qubits: expected {num_qubits}, got {loaded.num_qubits}"
        )
    if loaded.num_parameters != 0:
        errors.append(
            f"num_parameters: expected 0 (fully bound), got {loaded.num_parameters}"
        )
    if loaded.depth() != decomposed.depth():
        errors.append(
            f"depth: expected {decomposed.depth()}, got {loaded.depth()}"
        )

    if errors:
        print("\nV20 FAIL:")
        for e in errors:
            print(f"  {e}")
        raise AssertionError("QPY round-trip verification failed")

    print(f"\nV20 PASS: QPY round-trip verified")
    print(f"  num_qubits={loaded.num_qubits}")
    print(f"  depth={loaded.depth()}")
    print(f"  num_parameters={loaded.num_parameters} (bound — correct)")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export optimised VQE ansatz circuit to QPY format (V20 validation)"
    )
    parser.add_argument("result_json", help="Path to *_result.json")
    parser.add_argument(
        "--output", "-o", default=None,
        help="Output .qpy path (default: same dir as result JSON, {exp_id}_ansatz.qpy)"
    )
    args = parser.parse_args()

    path = export_qpy(args.result_json, args.output)
    print(f"\nDone: {path}")
