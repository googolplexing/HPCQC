#!/usr/bin/env python3
"""Fetch Q50 calibration data and convert to HPCQC format.

Connects to Q50 via FiQCI, fetches the latest calibration metrics,
saves raw + converted formats, and prints a summary.

Can also fetch a specific calibration set by ID (from a previous job).

Usage:
    sbatch tests/fetch_q50_calibration.sh

    # Or fetch a specific calibration set:
    sbatch tests/fetch_q50_calibration.sh <calibration_set_id>

Outputs:
    results/q50_calibration_raw_<date>.json    — raw FiQCI API response
    examples/q50_calibration_<date>.json       — HPCQC format (for noise models)

Based on FiQCI example:
    https://github.com/FiQCI/fiqci-examples/blob/main/scripts/get_calibration_data.py

CSC calibration docs:
    https://docs.csc.fi/computing/quantum-computing/running-quantum-jobs/#calibration-data
"""
import os
import sys
import json
import time
from datetime import datetime

import requests
from iqm.iqm_client import IQMClient
from iqm.qiskit_iqm import IQMProvider


def get_calibration_data(client, calibration_set_id=None):
    """Fetch calibration metrics from FiQCI API.

    Adapted from FiQCI/fiqci-examples/scripts/get_calibration_data.py
    """
    headers = {"User-Agent": client._iqm_server_client._signature}
    bearer_token = client._iqm_server_client._auth_header_callback()
    headers["Authorization"] = bearer_token

    server_client = client._iqm_server_client
    root_url = server_client.root_url
    quantum_computer = server_client.quantum_computer

    if calibration_set_id:
        url = f"{root_url}/api/devices/{quantum_computer}/calibration/metrics/{calibration_set_id}"
        print(f"  Fetching calibration set: {calibration_set_id}")
    else:
        url = f"{root_url}/api/devices/{quantum_computer}/calibration/metrics/latest"
        print(f"  Fetching latest calibration...")

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()


def convert_to_hpcqc_format(raw_data, device_name="Q50"):
    """Convert FiQCI raw calibration to HPCQC JSON format.

    VTT API response structure (confirmed April 7, 2026):
        {
            "metrics": {
                "t1_time": {
                    "QB1": {"unit": "s", "value": 3.87e-05, ...},
                    "QB2": {"unit": "s", "value": 2.86e-05, ...},
                    "statistics": {...},  # aggregate — skip
                    ...
                },
                "cz_irb_crf_crf_fidelity": {
                    "QB1__QB2": {"value": 0.976, ...},  # double underscore
                    ...
                },
                ...
            }
        }

    Structure is: metric_type → qubit_name → {unit, value, timestamp, uncertainty}
    CZ gate keys use double underscore: QB1__QB2 (not dash)
    We pivot to:  qubit_name → {metric_type: value}
    We normalize: QB1__QB2 → QB1-QB2 for HPCQC format

    HPCQC output format (used by IQMv2Adapter):
        {
            "calibration_set_id": str,
            "timestamp": str,
            "device": "Q50",
            "qubits": {
                "QB1": {"t1_us": float, "t2_us": float,
                        "readout_fidelity": float, "single_gate_error": float},
                ...
            },
            "two_qubit_gates": {
                "QB1-QB2": {"cz_fidelity": float, "cz_error": float},
                ...
            },
            "single_gate_time_ns": float,
            "cz_gate_time_ns": float
        }
    """
    cal_id = raw_data.get("calibration_set_id", "unknown")
    timestamp = raw_data.get(
        "calibration_set_end_timestamp",
        raw_data.get("timestamp", datetime.utcnow().isoformat()),
    )

    metrics = raw_data.get("metrics", {})

    print(f"\n  Raw response top-level keys: {list(raw_data.keys())}")
    print(f"  Metrics keys: {list(metrics.keys())}")

    # ── Pivot: metric_type → {qubit: {value,...}} to qubit → {metric: value} ──
    qubits_raw = {}
    gates_raw = {}

    for metric_name, qubit_values in metrics.items():
        if not isinstance(qubit_values, dict):
            continue
        for qubit_key, entry in qubit_values.items():
            # Skip aggregate statistics entries
            if qubit_key == "statistics":
                continue

            # Each entry is either a dict with "value" key, or a bare number
            if isinstance(entry, dict):
                val = entry.get("value")
            elif isinstance(entry, (int, float)):
                val = entry
            else:
                val = None

            # Two-qubit gate: QB1__QB2 (double underscore in VTT API)
            # Normalize to QB1-QB2 (single dash) for HPCQC format
            if "__" in qubit_key:
                normalized = qubit_key.replace("__", "-")
                gates_raw.setdefault(normalized, {})[metric_name] = val
            elif qubit_key.startswith("QB"):
                qubits_raw.setdefault(qubit_key, {})[metric_name] = val

    # ── Map VTT metric names → HPCQC fields ──
    #
    # VTT metric name                        → HPCQC field
    # t1_time (seconds)                      → t1_us (microseconds)
    # t2_time (seconds)                      → t2_us (microseconds)
    # t2_echo_time (seconds)                 → t2_echo_us (microseconds)
    # measure_ssro_constant_fidelity         → readout_fidelity
    # prx_rb_drag_crf_sx_fidelity            → 1.0 - single_gate_error
    # cz_irb_crf_crf_fidelity                → cz_fidelity

    qubits = {}
    for qb, m in sorted(qubits_raw.items(), key=_sort_qb):
        t1_s = m.get("t1_time")
        t2_s = m.get("t2_time")
        t2_echo_s = m.get("t2_echo_time")
        ro_fid = m.get("measure_ssro_constant_fidelity")
        gate_fid = m.get("prx_rb_drag_crf_sx_fidelity")

        qubits[qb] = {
            "t1_us": t1_s * 1e6 if t1_s is not None else None,
            "t2_us": t2_s * 1e6 if t2_s is not None else None,
            "t2_echo_us": t2_echo_s * 1e6 if t2_echo_s is not None else None,
            "readout_fidelity": ro_fid,
            "single_gate_error": (1.0 - gate_fid) if gate_fid is not None else None,
        }

    gates = {}
    for edge, m in sorted(gates_raw.items()):
        cz_fid = m.get("cz_irb_crf_crf_fidelity")
        if cz_fid is not None:
            gates[edge] = {
                "cz_fidelity": cz_fid,
                "cz_error": 1.0 - cz_fid,
            }

    print(f"  Parsed: {len(qubits)} qubits, {len(gates)} CZ gates")

    return {
        "calibration_set_id": cal_id,
        "timestamp": timestamp,
        "device": device_name,
        "notes": f"Fetched from FiQCI API on {datetime.utcnow().isoformat()}",
        "qubits": qubits,
        "two_qubit_gates": gates,
        "single_gate_time_ns": 20,
        "cz_gate_time_ns": 100,
    }


def _sort_qb(item):
    """Sort QB1, QB2, ..., QB54 numerically (not lexicographically)."""
    name = item[0]
    try:
        return int(name.replace("QB", ""))
    except ValueError:
        return 9999


def print_summary(cal):
    """Print calibration summary statistics."""
    import numpy as np

    qubits = cal["qubits"]
    gates = cal["two_qubit_gates"]

    t1s = [q["t1_us"] for q in qubits.values() if q.get("t1_us") is not None]
    t2s = [q["t2_us"] for q in qubits.values() if q.get("t2_us") is not None]
    ros = [q["readout_fidelity"] for q in qubits.values() if q.get("readout_fidelity") is not None]
    ges = [q["single_gate_error"] for q in qubits.values() if q.get("single_gate_error") is not None]
    czs = [g["cz_fidelity"] for g in gates.values() if g.get("cz_fidelity") is not None]

    print(f"\n{'═' * 60}")
    print(f"  Q50 CALIBRATION SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Calibration ID: {cal['calibration_set_id']}")
    print(f"  Timestamp:      {cal['timestamp']}")
    print(f"  Device:         {cal['device']}")
    print(f"  Qubits:         {len(qubits)}")
    print(f"  CZ gates:       {len(gates)}")
    print()

    if t1s:
        print(f"  T1 (µs):         min={min(t1s):6.1f}  median={np.median(t1s):6.1f}  max={max(t1s):6.1f}")
    if t2s:
        print(f"  T2 (µs):         min={min(t2s):6.1f}  median={np.median(t2s):6.1f}  max={max(t2s):6.1f}")
    if ros:
        print(f"  Readout fid:     min={min(ros):.4f}  median={np.median(ros):.4f}  max={max(ros):.4f}")
    if ges:
        print(f"  1q gate error:   min={min(ges):.5f}  median={np.median(ges):.5f}  max={max(ges):.5f}")
    if czs:
        print(f"  CZ fidelity:     min={min(czs):.4f}  median={np.median(czs):.4f}  max={max(czs):.4f}")

    # Flag problem qubits
    if ros:
        bad_ro = [(n, q["readout_fidelity"]) for n, q in qubits.items()
                  if q.get("readout_fidelity") is not None and q["readout_fidelity"] < 0.90]
        if bad_ro:
            print(f"\n  ⚠ Low readout fidelity (<90%):")
            for name, fid in sorted(bad_ro, key=lambda x: x[1]):
                print(f"    {name}: {fid:.4f}")

    if t2s:
        bad_t2 = [(n, q["t2_us"]) for n, q in qubits.items()
                  if q.get("t2_us") is not None and q["t2_us"] < 2.0]
        if bad_t2:
            print(f"\n  ⚠ Very short T2 (<2 µs):")
            for name, val in sorted(bad_t2, key=lambda x: x[1]):
                print(f"    {name}: {val:.2f} µs")

    if czs:
        bad_cz = [(n, g["cz_fidelity"]) for n, g in gates.items()
                  if g.get("cz_fidelity") is not None and g["cz_fidelity"] < 0.95]
        if bad_cz:
            print(f"\n  ⚠ Low CZ fidelity (<95%):")
            for name, fid in sorted(bad_cz, key=lambda x: x[1])[:10]:
                print(f"    {name}: {fid:.4f}")
            if len(bad_cz) > 10:
                print(f"    ... and {len(bad_cz) - 10} more")

    print(f"{'═' * 60}")


def main():
    # Connect to Q50
    DEVICE_CORTEX_URL = os.getenv("Q50_CORTEX_URL")
    if not DEVICE_CORTEX_URL:
        raise EnvironmentError("Q50_CORTEX_URL not set — are you on q_fiqci partition?")

    provider = IQMProvider(DEVICE_CORTEX_URL, quantum_computer="q50")
    backend = provider.get_backend()

    print(f"Connected to Q50 ({backend.num_qubits} qubits)")

    # Optional: fetch specific calibration set ID from command line
    cal_set_id = sys.argv[1] if len(sys.argv) > 1 else None

    # Fetch calibration data
    raw_data = get_calibration_data(backend.client, cal_set_id)

    # Save raw response
    date_str = datetime.utcnow().strftime("%Y%m%d")
    cal_id = raw_data.get("calibration_set_id", "unknown")
    cal_id_short = cal_id[:8] if cal_id != "unknown" else "unknown"
    os.makedirs("results", exist_ok=True)
    raw_path = f"results/q50_calibration_raw_{date_str}_{cal_id_short}.json"
    with open(raw_path, "w") as f:
        json.dump(raw_data, f, indent=2, default=str)
    print(f"  Raw data saved: {raw_path}")

    # Convert to HPCQC format
    hpcqc_cal = convert_to_hpcqc_format(raw_data)

    # Save HPCQC format
    os.makedirs("examples", exist_ok=True)
    hpcqc_path = f"examples/q50_calibration_{date_str}_{cal_id_short}.json"
    with open(hpcqc_path, "w") as f:
        json.dump(hpcqc_cal, f, indent=2)
    print(f"  HPCQC format saved: {hpcqc_path}")

    # Print summary
    print_summary(hpcqc_cal)

    # Print backend info
    try:
        edges = backend.coupling_map.get_edges()
        print(f"\n  Backend coupling map: {len(edges)} edges")
    except Exception:
        print(f"\n  Backend coupling map: available (could not count edges)")
    print(f"  Native ops: {backend.operation_names}")


if __name__ == "__main__":
    main()
