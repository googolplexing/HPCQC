# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Build Aer noise models from Q50 calibration data.

Creates a NoiseModel with:
  - Depolarizing errors on single-qubit gates (from single_gate_error)
  - Depolarizing errors on two-qubit gates (from cz_error)
  - Thermal relaxation (from T1, T2, gate times)
  - Readout errors (from readout_fidelity)

Usage:
    from lumi_hpc_qc.backends.noise_model import build_noise_model
    noise_model = build_noise_model("examples/q50_calibration_20260326.json", num_qubits=8)
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
        qiskit_aer.noise.NoiseModel configured with Q50 noise characteristics.
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error, ReadoutError
    import numpy as np

    with open(calibration_path) as f:
        cal = json.load(f)

    qubits_data = cal.get("qubits", {})
    gates_data = cal.get("two_qubit_gates", {})
    sg_time_ns = cal.get("single_gate_time_ns", 40)
    cz_time_ns = cal.get("cz_gate_time_ns", 100)

    # Select best qubits by readout fidelity
    sorted_qubits = sorted(
        qubits_data.items(),
        key=lambda x: x[1]["readout_fidelity"],
        reverse=True,
    )
    selected = sorted_qubits[:num_qubits]
    qubit_names = [q[0] for q in selected]

    print(f"  Noise model: {num_qubits} qubits from Q50 calibration")
    print(f"  Physical qubits: {', '.join(qubit_names)}")
    ro_vals = ', '.join(f'{q[1]["readout_fidelity"]:.3f}' for q in selected)
    print(f"  Readout fidelities: {ro_vals}")

    noise_model = NoiseModel()

    # Single-qubit gate errors (depolarizing + thermal relaxation)
    for i, (qname, qdata) in enumerate(selected):
        # Depolarizing error
        sg_err = qdata.get("single_gate_error", 0.001)
        if sg_err > 0:
            dep_err = depolarizing_error(sg_err, 1)
            noise_model.add_quantum_error(dep_err, ['rx', 'ry', 'rz', 'x', 'h', 'sx'], [i])

        # Thermal relaxation
        t1 = qdata.get("t1_us", 40.0) * 1e3  # convert to ns
        t2 = qdata.get("t2_us", 20.0) * 1e3
        if t2 > 2 * t1:
            t2 = 2 * t1  # physical constraint
        if t1 > 0 and t2 > 0:
            therm_err = thermal_relaxation_error(t1, t2, sg_time_ns)
            noise_model.add_quantum_error(therm_err, ['rx', 'ry', 'rz', 'x', 'h', 'sx'], [i])

        # Readout error
        ro_fid = qdata.get("readout_fidelity", 0.97)
        p0_given_1 = 1 - ro_fid  # P(measure 0 | state is 1)
        p1_given_0 = (1 - ro_fid) * 0.5  # P(measure 1 | state is 0), typically lower
        ro_err = ReadoutError([[1 - p1_given_0, p1_given_0], [p0_given_1, ro_fid]])
        noise_model.add_readout_error(ro_err, [i])

    # Two-qubit gate errors
    # Map physical qubit pairs to logical indices
    name_to_idx = {qname: i for i, (qname, _) in enumerate(selected)}

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

        # Thermal relaxation during CZ gate
        for qubit_idx in [i, j]:
            qdata = selected[qubit_idx][1]
            t1 = qdata.get("t1_us", 40.0) * 1e3
            t2 = qdata.get("t2_us", 20.0) * 1e3
            if t2 > 2 * t1:
                t2 = 2 * t1
            if t1 > 0 and t2 > 0:
                therm_err_cz = thermal_relaxation_error(t1, t2, cz_time_ns)
                noise_model.add_quantum_error(therm_err_cz, ['cx', 'cz'], [qubit_idx])

    avg_ro = np.mean([q[1]["readout_fidelity"] for q in selected])
    avg_sg = np.mean([q[1]["single_gate_error"] for q in selected])
    print(f"  Avg readout fidelity: {avg_ro:.4f}")
    print(f"  Avg single-gate error: {avg_sg:.5f}")
    print(f"  Two-qubit gate pairs mapped: {sum(1 for g in gates_data if all(p in name_to_idx for p in g.split('-')))}")

    return noise_model
