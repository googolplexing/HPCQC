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

    HPCQC format (used by IQMv2Adapter):
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
    # Extract calibration set ID and timestamp
    cal_id = raw_data.get("calibration_set_id", "unknown")
    timestamp = raw_data.get("timestamp", datetime.utcnow().isoformat())

    # The raw API response structure varies — print keys to understand it
    print(f"\n  Raw response top-level keys: {list(raw_data.keys())}")

    # Try to extract qubit metrics
    qubits = {}
    gates = {}

    # FiQCI metrics are typically nested under 'metrics' or directly
    metrics = raw_data.get("metrics", raw_data)

    if isinstance(metrics, dict):
        print(f"  Metrics keys: {list(metrics.keys())[:20]}")

    # Strategy: iterate through metrics looking for qubit and gate data
    # FiQCI format uses keys like "QB1", "QB1-QB2" with nested metric dicts
    for key, value in metrics.items():
        if not isinstance(value, dict):
            continue

        # Single qubit metrics (key looks like "QB1", "QB2", etc.)
        if key.startswith("QB") and "-" not in key:
            t1 = None
            t2 = None
            ro_fidelity = None
            gate_error = None

            # Try various known FiQCI metric key names
            for metric_key, metric_val in value.items():
                val = metric_val if isinstance(metric_val, (int, float)) else None
                if val is None and isinstance(metric_val, dict):
                    val = metric_val.get("value", metric_val.get("fidelity"))

                mk = metric_key.lower()
                if "t1" in mk and "time" in mk:
                    t1 = val * 1e6 if val and val < 1 else val  # s → µs
                elif "t2" in mk and "echo" not in mk and "time" in mk:
                    t2 = val * 1e6 if val and val < 1 else val
                elif "ssro" in mk and "fidelity" in mk:
                    ro_fidelity = val
                elif "prx" in mk and "fidelity" in mk:
                    gate_error = 1.0 - val if val else None

            qubits[key] = {
                "t1_us": t1 or 30.0,
                "t2_us": t2 or 10.0,
                "readout_fidelity": ro_fidelity or 0.95,
                "single_gate_error": gate_error or 0.002,
            }

        # Two-qubit gate metrics (key looks like "QB1-QB2")
        elif "-" in key and key.split("-")[0].startswith("QB"):
            cz_fidelity = None
            for metric_key, metric_val in value.items():
                val = metric_val if isinstance(metric_val, (int, float)) else None
                if val is None and isinstance(metric_val, dict):
                    val = metric_val.get("value", metric_val.get("fidelity"))

                mk = metric_key.lower()
                if "cz" in mk and ("fidelity" in mk or "irb" in mk):
                    cz_fidelity = val

            if cz_fidelity is not None:
                gates[key] = {
                    "cz_fidelity": cz_fidelity,
                    "cz_error": 1.0 - cz_fidelity,
                }

    hpcqc_cal = {
        "calibration_set_id": cal_id,
        "timestamp": timestamp,
        "device": device_name,
        "notes": f"Fetched from FiQCI API on {datetime.utcnow().isoformat()}",
        "qubits": qubits,
        "two_qubit_gates": gates,
        "single_gate_time_ns": 20,
        "cz_gate_time_ns": 100,
    }

    return hpcqc_cal


def print_summary(cal):
    """Print calibration summary statistics."""
    import numpy as np

    qubits = cal["qubits"]
    gates = cal["two_qubit_gates"]

    t1s = [q["t1_us"] for q in qubits.values() if q["t1_us"]]
    t2s = [q["t2_us"] for q in qubits.values() if q["t2_us"]]
    ros = [q["readout_fidelity"] for q in qubits.values() if q["readout_fidelity"]]
    czs = [g["cz_fidelity"] for g in gates.values() if g["cz_fidelity"]]

    print(f"\n═══════════════════════════════════════════════════════")
    print(f"  Q50 CALIBRATION SUMMARY")
    print(f"═══════════════════════════════════════════════════════")
    print(f"  Calibration ID: {cal['calibration_set_id']}")
    print(f"  Timestamp:      {cal['timestamp']}")
    print(f"  Device:         {cal['device']}")
    print(f"  Qubits:         {len(qubits)}")
    print(f"  CZ gates:       {len(gates)}")
    print()

    if t1s:
        print(f"  T1 (µs):     min={min(t1s):.1f}  median={np.median(t1s):.1f}  max={max(t1s):.1f}")
    if t2s:
        print(f"  T2 (µs):     min={min(t2s):.1f}  median={np.median(t2s):.1f}  max={max(t2s):.1f}")
    if ros:
        print(f"  Readout:     min={min(ros):.4f}  median={np.median(ros):.4f}  max={max(ros):.4f}")
    if czs:
        print(f"  CZ fidelity: min={min(czs):.4f}  median={np.median(czs):.4f}  max={max(czs):.4f}")

    # Flag problem qubits
    if ros:
        bad_ro = [(n, q["readout_fidelity"]) for n, q in qubits.items()
                  if q["readout_fidelity"] and q["readout_fidelity"] < 0.90]
        if bad_ro:
            print(f"\n  ⚠ Low readout fidelity (<90%):")
            for name, fid in sorted(bad_ro, key=lambda x: x[1]):
                print(f"    {name}: {fid:.4f}")

    if czs:
        bad_cz = [(n, g["cz_fidelity"]) for n, g in gates.items()
                  if g["cz_fidelity"] and g["cz_fidelity"] < 0.95]
        if bad_cz:
            print(f"\n  ⚠ Low CZ fidelity (<95%):")
            for name, fid in sorted(bad_cz, key=lambda x: x[1])[:10]:
                print(f"    {name}: {fid:.4f}")
            if len(bad_cz) > 10:
                print(f"    ... and {len(bad_cz) - 10} more")

    print(f"═══════════════════════════════════════════════════════")


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
    print(f"\n  Backend coupling map: {len(backend.coupling_map)} edges")
    print(f"  Native ops: {backend.operation_names}")


if __name__ == "__main__":
    main()
