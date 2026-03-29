# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Build Aer noise models from Q50 calibration data.

Fixes applied:
  C4: Single-qubit gate errors use ONLY depolarizing (from RB data).
      Thermal relaxation is NOT added on top, since RB-measured
      single_gate_error already includes coherence contributions.
      Thermal relaxation is applied separately for idle periods
      during two-qubit gates on spectator qubits (not yet implemented).
  C5: Readout error uses symmetric model: P(0|1) = P(1|0) = (1-fidelity)/2.
      The calibration JSON provides only a single readout_fidelity number,
      not the asymmetric rates. Using an arbitrary 0.5 factor for P(1|0)
      creates false precision. Symmetric is honest about available data.
  C1: Returns the coupling map alongside the noise model so the Aer
      backend can transpile circuits to Q50 topology before simulation.

Usage:
    from lumi_hpc_qc.backends.noise_model import build_noise_model
    noise_model, coupling_map = build_noise_model("examples/q50_calibration_20260326.json", 8)
"""

from __future__ import annotations

import json
from pathlib import Path


def build_noise_model(calibration_path: str, num_qubits: int = 8):
    """Build an Aer NoiseModel from Q50 calibration JSON.

    Maps logical qubits 0..n-1 to the best available physical qubits
    from the calibration data (sorted by readout fidelity descending).

    Args:
        calibration_path: Path to calibration JSON file.
        num_qubits: Number of qubits in the circuit.

    Returns:
        (noise_model, coupling_map) — NoiseModel and CouplingMap for
        topology-aware simulation. coupling_map may be None if the
        calibration data doesn't contain topology information.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    from qiskit.transpiler import CouplingMap
    import numpy as np

    with open(calibration_path) as f:
        cal = json.load(f)

    qubits_data = cal.get("qubits", {})
    gates_data = cal.get("two_qubit_gates", {})

    # Select best qubits by readout fidelity
    sorted_qubits = sorted(
        qubits_data.items(),
        key=lambda x: x[1]["readout_fidelity"],
        reverse=True,
    )
    selected = sorted_qubits[:num_qubits]
    qubit_names = [q[0] for q in selected]
    name_to_idx = {qname: i for i, (qname, _) in enumerate(selected)}

    ro_vals = ', '.join(f'{q[1]["readout_fidelity"]:.3f}' for q in selected)
    print(f"  Noise model: {num_qubits} qubits from Q50 calibration")
    print(f"  Physical qubits: {', '.join(qubit_names)}")
    print(f"  Readout fidelities: {ro_vals}")

    noise_model = NoiseModel()

    # ── Single-qubit gate errors ──
    # C4 FIX: Use ONLY depolarizing error from RB data.
    # RB-measured single_gate_error already includes T1/T2 contributions
    # during the gate time. Adding thermal_relaxation_error on top
    # double-counts the coherence contribution.
    for i, (qname, qdata) in enumerate(selected):
        sg_err = qdata.get("single_gate_error", 0.001)
        if sg_err > 0:
            dep_err = depolarizing_error(sg_err, 1)
            noise_model.add_quantum_error(
                dep_err, ['rx', 'ry', 'rz', 'x', 'h', 'sx'], [i]
            )

        # C5 FIX: Symmetric readout error model.
        # Calibration provides only readout_fidelity (single number).
        # Asymmetric rates P(0|1) and P(1|0) are not available.
        # Symmetric model: p_error = (1 - fidelity) / 2 for both directions.
        ro_fid = qdata.get("readout_fidelity", 0.97)
        p_error = (1 - ro_fid) / 2
        ro_err = ReadoutError([
            [1 - p_error, p_error],
            [p_error, 1 - p_error],
        ])
        noise_model.add_readout_error(ro_err, [i])

    # ── Two-qubit gate errors ──
    # Use depolarizing error from CZ fidelity data
    coupling_edges = []
    for gate_pair, gate_data in gates_data.items():
        parts = gate_pair.split("-")
        if len(parts) != 2:
            continue
        q1_name, q2_name = parts
        if q1_name not in name_to_idx or q2_name not in name_to_idx:
            continue

        i, j = name_to_idx[q1_name], name_to_idx[q2_name]
        cz_err = gate_data.get("cz_error", 0.005)

        if cz_err > 0:
            dep_err_2q = depolarizing_error(cz_err, 2)
            noise_model.add_quantum_error(dep_err_2q, ['cx', 'cz'], [i, j])
            noise_model.add_quantum_error(dep_err_2q, ['cx', 'cz'], [j, i])

        # Record edge for coupling map
        coupling_edges.append([i, j])
        coupling_edges.append([j, i])

    # Build coupling map from calibration topology
    # C1 FIX: return coupling map so circuits can be transpiled
    coupling_map = None
    if coupling_edges:
        coupling_map = CouplingMap(coupling_edges)

    avg_ro = np.mean([q[1]["readout_fidelity"] for q in selected])
    avg_sg = np.mean([q[1]["single_gate_error"] for q in selected])
    mapped_pairs = sum(
        1 for g in gates_data
        if all(p in name_to_idx for p in g.split('-'))
    )
    print(f"  Avg readout fidelity: {avg_ro:.4f}")
    print(f"  Avg single-gate error: {avg_sg:.5f}")
    print(f"  Two-qubit gate pairs mapped: {mapped_pairs}")
    if coupling_map:
        print(f"  Coupling map edges: {len(coupling_edges) // 2} (bidirectional)")

    return noise_model, coupling_map
