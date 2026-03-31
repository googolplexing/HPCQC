# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""QPY export — serialise the optimised ansatz circuit from a completed VQE run.

V20 validation: QPY file must be loadable by qpy.load() and the loaded
circuit must have the correct num_qubits and num_parameters.

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
    """Rebuild ansatz from result JSON and serialise to QPY.

    Args:
        result_json_path: Path to *_result.json from a completed VQE run.
        output_path: Output .qpy path. Defaults to same dir as result JSON.

    Returns:
        Path to the written QPY file.
    """
    # ── Load result JSON ──────────────────────────────────────────────
    with open(result_json_path) as f:
        result = json.load(f)

    exp_id      = result["experiment_id"]
    config      = result["config"]
    convergence = result.get("convergence", {})
    iterations  = result.get("iterations", [])

    # ── Extract best parameters ───────────────────────────────────────
    best_params = None
    for it in reversed(iterations):
        if it.get("is_best"):
            best_params = it["parameters"]
            break
    if best_params is None and iterations:
        best_params = iterations[-1]["parameters"]
    if best_params is None:
        raise ValueError("No parameter data found in result JSON")

    import numpy as np
    best_params = np.array(best_params)

    # ── Rebuild ansatz circuit ────────────────────────────────────────
    from lumi_hpc_qc.plugins.registry import PluginRegistry
    registry = PluginRegistry()

    num_qubits  = config["num_qubits"]
    ansatz_name = config["ansatz"]
    ansatz_params = config.get("ansatz_params", {})

    ansatz_builder = registry.get_ansatz(ansatz_name)
    ansatz = ansatz_builder.build(num_qubits, ansatz_params)

    print(f"Ansatz: {ansatz_name}, {num_qubits} qubits, "
          f"{ansatz.num_parameters} parameters")
    print(f"Best energy: {convergence.get('best_energy')}")
    print(f"Best relative error: {convergence.get('relative_error_pct', '?')}%")

    # ── Bind best parameters ──────────────────────────────────────────
    bound_circuit = ansatz.assign_parameters(best_params)
    print(f"Bound circuit depth: {bound_circuit.depth()}")

    # ── Decompose for Aer (same as workflow) ─────────────────────────
    from lumi_hpc_qc.backends.aer_gpu import decompose_for_aer
    decomposed = decompose_for_aer(bound_circuit)
    print(f"Decomposed circuit depth: {decomposed.depth()}, "
          f"gates: {decomposed.count_ops()}")

    # ── Serialise to QPY ──────────────────────────────────────────────
    from qiskit.qpy import dump as qpy_dump

    if output_path is None:
        result_dir  = os.path.dirname(os.path.abspath(result_json_path))
        output_path = os.path.join(result_dir, f"{exp_id}_ansatz.qpy")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    with open(output_path, "wb") as f:
        qpy_dump(decomposed, f)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"QPY written: {output_path} ({size_kb:.1f} KB)")

    # ── V20 verification: load back and check ────────────────────────
    from qiskit.qpy import load as qpy_load

    with open(output_path, "rb") as f:
        loaded = qpy_load(f)

    loaded_circuit = loaded[0]
    assert loaded_circuit.num_qubits == num_qubits, (
        f"V20 FAIL: num_qubits mismatch: "
        f"expected {num_qubits}, got {loaded_circuit.num_qubits}"
    )
    # Bound circuit has 0 free parameters (all bound)
    assert loaded_circuit.num_parameters == 0, (
        f"V20 FAIL: expected 0 free parameters (bound circuit), "
        f"got {loaded_circuit.num_parameters}"
    )
    assert loaded_circuit.depth() == decomposed.depth(), (
        f"V20 FAIL: depth mismatch after round-trip: "
        f"expected {decomposed.depth()}, got {loaded_circuit.depth()}"
    )

    print(f"V20 PASS: QPY round-trip verified — "
          f"num_qubits={loaded_circuit.num_qubits}, "
          f"depth={loaded_circuit.depth()}, "
          f"num_parameters={loaded_circuit.num_parameters} (bound)")

    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export optimised VQE ansatz circuit to QPY format (V20)"
    )
    parser.add_argument("result_json", help="Path to *_result.json")
    parser.add_argument("--output", "-o", default=None,
                        help="Output .qpy path (default: same dir as result JSON)")
    args = parser.parse_args()

    path = export_qpy(args.result_json, args.output)
    print(f"\nDone: {path}")
