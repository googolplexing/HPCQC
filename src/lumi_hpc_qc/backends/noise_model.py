# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Build Aer noise models from Q50 calibration data.

Phase B: Parameterized noise model with configurable channels.

Noise channels (all default to True for backward compatibility):
  - single_qubit_depolarizing: RB-measured gate error on 1q gates
  - two_qubit_depolarizing: CZ fidelity-based error on 2q gates
  - t1_relaxation: T1 amplitude damping during idle periods
  - t2_dephasing: T2 dephasing during idle periods
  - readout_error: Symmetric measurement error from readout fidelity

Fixes preserved:
  C4: Depolarizing (from RB) on gate ops; thermal on idle ops. No double-counting.
  C5: Symmetric readout error: (1-fid)/2 both directions.
  C1: Returns coupling map alongside noise model for topology-aware sim.

Usage:
    from lumi_hpc_qc.backends.noise_model import build_noise_model, extract_coupling_map

    # Full noise (backward compatible):
    noise_model, coupling_map = build_noise_model("calibration.json", 8)

    # Individual channel isolation:
    noise_model, coupling_map = build_noise_model(
        "calibration.json", 8,
        noise_channels={"readout_error": True}  # others default False
    )

    # Topology only (no noise):
    coupling_map = extract_coupling_map("calibration.json", 8)
"""

from __future__ import annotations

import json
from typing import Any


_ALL_CHANNELS = {
    "single_qubit_depolarizing": True,
    "two_qubit_depolarizing": True,
    "t1_relaxation": True,
    "t2_dephasing": True,
    "readout_error": True,
}


def _load_calibration(calibration_path: str) -> dict:
    with open(calibration_path) as f:
        return json.load(f)


def _select_qubits(cal: dict, num_qubits: int) -> list[tuple[str, dict]]:
    """Select best connected subgraph of qubits by readout fidelity.

    Starts from the highest-fidelity qubit and greedily adds the
    highest-fidelity neighbor until num_qubits are selected. This
    ensures the selected qubits form a connected subgraph — required
    for both coupling map extraction and noise model building.
    """
    qubits_data = cal.get("qubits", {})
    gates_data = cal.get("two_qubit_gates", {})

    if num_qubits >= len(qubits_data):
        return sorted(qubits_data.items(),
                       key=lambda x: x[1]["readout_fidelity"], reverse=True)

    # Build adjacency from gate pairs
    adj: dict[str, set[str]] = {q: set() for q in qubits_data}
    for gate_pair in gates_data:
        parts = gate_pair.split("-")
        if len(parts) != 2:
            continue
        q1, q2 = parts
        if q1 in adj and q2 in adj:
            adj[q1].add(q2)
            adj[q2].add(q1)

    # Greedy: start from highest-fidelity connected qubit, expand by best neighbor
    sorted_by_fid = sorted(qubits_data.items(),
                            key=lambda x: x[1]["readout_fidelity"], reverse=True)

    for start_name, start_data in sorted_by_fid:
        if not adj[start_name]:
            continue
        selected = {start_name: start_data}
        frontier = set(adj[start_name])

        while len(selected) < num_qubits and frontier:
            best = max(frontier, key=lambda q: qubits_data[q]["readout_fidelity"])
            selected[best] = qubits_data[best]
            frontier.discard(best)
            for nb in adj[best]:
                if nb not in selected:
                    frontier.add(nb)

        if len(selected) >= num_qubits:
            return list(selected.items())[:num_qubits]

    # Fallback: return top-N by fidelity (may be disconnected)
    return sorted_by_fid[:num_qubits]


def _resolve_channels(noise_channels: dict | None) -> dict:
    """If None: all active. If dict: only True channels active, missing default False."""
    if noise_channels is None:
        return dict(_ALL_CHANNELS)
    return {key: noise_channels.get(key, False) for key in _ALL_CHANNELS}


def _extract_edges(cal: dict, name_to_idx: dict) -> list:
    gates_data = cal.get("two_qubit_gates", {})
    edges = []
    for gate_pair in gates_data:
        parts = gate_pair.split("-")
        if len(parts) != 2:
            continue
        q1, q2 = parts
        if q1 not in name_to_idx or q2 not in name_to_idx:
            continue
        i, j = name_to_idx[q1], name_to_idx[q2]
        edges.append([i, j])
        edges.append([j, i])
    return edges


def extract_coupling_map(calibration_path: str, num_qubits: int = 8):
    """Extract coupling map from calibration without building noise model.

    Used by topology-noiseless mode.
    """
    from qiskit.transpiler import CouplingMap

    cal = _load_calibration(calibration_path)
    selected = _select_qubits(cal, num_qubits)
    name_to_idx = {qname: i for i, (qname, _) in enumerate(selected)}
    edges = _extract_edges(cal, name_to_idx)

    if edges:
        print(f"  Coupling map: {num_qubits} qubits, {len(edges) // 2} edges (topology only)")
        return CouplingMap(edges)
    return None


def build_noise_model(
    calibration_path: str,
    num_qubits: int = 8,
    noise_channels: dict | None = None,
):
    """Build an Aer NoiseModel from Q50 calibration JSON.

    Args:
        calibration_path: Path to calibration JSON file.
        num_qubits: Number of qubits in the circuit.
        noise_channels: Dict of channel_name -> bool. None = all active.

    Returns:
        (noise_model, coupling_map)
    """
    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    from qiskit_aer.noise.errors.standard_errors import thermal_relaxation_error
    from qiskit.transpiler import CouplingMap
    import numpy as np

    channels = _resolve_channels(noise_channels)
    cal = _load_calibration(calibration_path)
    selected = _select_qubits(cal, num_qubits)
    name_to_idx = {qname: i for i, (qname, _) in enumerate(selected)}
    qubit_names = [q[0] for q in selected]
    gates_data = cal.get("two_qubit_gates", {})
    cz_time_ns = cal.get("cz_gate_time_ns", 100)

    active_str = ", ".join(k for k, v in channels.items() if v)
    print(f"  Noise model: {num_qubits} qubits from Q50 calibration")
    print(f"  Physical qubits: {', '.join(qubit_names)}")
    print(f"  Active channels: {active_str or 'none'}")

    noise_model = NoiseModel()

    # ── Single-qubit depolarizing (from RB) ──
    if channels["single_qubit_depolarizing"]:
        for i, (qname, qdata) in enumerate(selected):
            sg_err = qdata.get("single_gate_error", 0.001)
            if sg_err > 0:
                dep_err = depolarizing_error(sg_err, 1)
                noise_model.add_quantum_error(
                    dep_err, ['rx', 'ry', 'rz', 'x', 'h', 'sx'], [i]
                )

    # ── Two-qubit depolarizing (from CZ fidelity) ──
    coupling_edges = _extract_edges(cal, name_to_idx)
    if channels["two_qubit_depolarizing"]:
        for gate_pair, gate_data in gates_data.items():
            parts = gate_pair.split("-")
            if len(parts) != 2:
                continue
            q1, q2 = parts
            if q1 not in name_to_idx or q2 not in name_to_idx:
                continue
            i, j = name_to_idx[q1], name_to_idx[q2]
            cz_err = gate_data.get("cz_error", 0.005)
            if cz_err > 0:
                dep_err_2q = depolarizing_error(cz_err, 2)
                noise_model.add_quantum_error(dep_err_2q, ['cx', 'cz'], [i, j])
                noise_model.add_quantum_error(dep_err_2q, ['cx', 'cz'], [j, i])

    # ── T1/T2 thermal relaxation (idle time) ──
    # Applied to 'id' and 'delay' — NOT gate ops (avoids C4 double-counting).
    # Uniform-layer approximation: idle time ≈ CZ gate time.
    # If Aer doesn't apply noise to implicit idle qubits, circuits may need
    # explicit scheduling (PadDelay) before noisy simulation.
    if channels["t1_relaxation"] or channels["t2_dephasing"]:
        for i, (qname, qdata) in enumerate(selected):
            t1_ns = qdata.get("t1_us", 50.0) * 1e3
            t2_ns = qdata.get("t2_us", 20.0) * 1e3

            if not channels["t1_relaxation"]:
                t1_ns = float('inf')
            if not channels["t2_dephasing"]:
                t2_ns = float('inf')
            if t2_ns > 2 * t1_ns:
                t2_ns = 2 * t1_ns

            if t1_ns == float('inf') and t2_ns == float('inf'):
                continue

            thermal_err = thermal_relaxation_error(t1_ns, t2_ns, cz_time_ns)
            noise_model.add_quantum_error(thermal_err, ['id', 'delay'], [i])

    # ── Readout error (C5: symmetric) ──
    if channels["readout_error"]:
        for i, (qname, qdata) in enumerate(selected):
            ro_fid = qdata.get("readout_fidelity", 0.97)
            p_error = (1 - ro_fid) / 2
            ro_err = ReadoutError([
                [1 - p_error, p_error],
                [p_error, 1 - p_error],
            ])
            noise_model.add_readout_error(ro_err, [i])

    # ── Coupling map ──
    coupling_map = CouplingMap(coupling_edges) if coupling_edges else None

    # ── Summary ──
    avg_ro = np.mean([q[1]["readout_fidelity"] for q in selected])
    avg_sg = np.mean([q[1]["single_gate_error"] for q in selected])
    mapped_pairs = sum(
        1 for g in gates_data if all(p in name_to_idx for p in g.split('-'))
    )
    print(f"  Avg readout fidelity: {avg_ro:.4f}")
    print(f"  Avg single-gate error: {avg_sg:.5f}")
    print(f"  Mapped 2q pairs: {mapped_pairs}")
    if coupling_map:
        print(f"  Coupling map edges: {len(coupling_edges) // 2}")

    return noise_model, coupling_map


def get_noise_config_metadata(
    calibration_path: str,
    num_qubits: int,
    noise_channels: dict | None,
    coupling_map_source: str,
) -> dict:
    """Build noise_config metadata for ExperimentRecord."""
    channels = _resolve_channels(noise_channels)
    cal = _load_calibration(calibration_path)
    selected = _select_qubits(cal, num_qubits)
    n = len(selected)

    return {
        "noise_model_file": calibration_path,
        "channels_active": channels,
        "coupling_map_source": coupling_map_source,
        "num_qubits_mapped": num_qubits,
        "qubit_assignment": [q[0] for q in selected],
        "calibration_summary": {
            "avg_readout_fidelity": sum(q[1]["readout_fidelity"] for q in selected) / n,
            "avg_single_gate_error": sum(q[1]["single_gate_error"] for q in selected) / n,
            "avg_t1_us": sum(q[1].get("t1_us", 50) for q in selected) / n,
            "avg_t2_us": sum(q[1].get("t2_us", 20) for q in selected) / n,
        },
    }
