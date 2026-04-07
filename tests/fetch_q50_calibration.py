#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Fetch Q50 calibration data, architecture, and convert to HPCQC format.

Connects to Q50 via FiQCI, fetches three data sources from VTT QX API:
  1. Calibration metrics (T1, T2, gate fidelities, readout fidelities)
  2. Static quantum architecture (qubit connectivity, native operations)
  3. Dynamic quantum architecture (per-calibration active qubits/gates)

Saves raw API responses for provenance and produces a single HPCQC-format
calibration JSON that contains both metrics AND topology.

Previous versions inferred connectivity from two_qubit_gates keys. This
version fetches the authoritative qubit_connectivity from the QX API
architecture endpoint, ensuring edges that exist physically but lack
calibration data in a particular cycle are not invisible to the placement
solver.

Usage:
    sbatch tests/fetch_q50_calibration.sh
    sbatch tests/fetch_q50_calibration.sh <calibration_set_id>

Outputs:
    results/q50_calibration_raw_<date>.json     -- raw metrics response
    results/q50_architecture_<date>.json        -- raw static architecture
    results/q50_dynamic_arch_<date>.json        -- raw dynamic architecture
    examples/q50_calibration_<date>.json        -- HPCQC format (metrics + topology)

VTT QX API version: qx-v0.6.7
Based on FiQCI example: https://github.com/FiQCI/fiqci-examples
CSC docs: https://docs.csc.fi/computing/quantum-computing/running-quantum-jobs/
"""
import os
import sys
import json
from datetime import datetime

from iqm.qiskit_iqm import IQMProvider


def convert_to_hpcqc_format(metrics_data, architecture=None,
                            dynamic_architecture=None, device_name="Q50"):
    """Convert VTT QX API responses to a single HPCQC calibration JSON.

    Combines calibration metrics (fidelities, coherence times) with
    quantum architecture (connectivity, native operations) into the
    format consumed by IQMv2Adapter.

    Args:
        metrics_data: Response from /api/devices/{device}/calibration/metrics/
        architecture: Response from /api/devices/{device}/quantum-architecture
        dynamic_architecture: Response from dynamic-quantum-architecture endpoint
        device_name: Device identifier for the output file.

    Returns:
        HPCQC calibration dict with keys: calibration_set_id, timestamp,
        device, adapter, qubits, two_qubit_gates, qubit_connectivity,
        active_qubits, native_operations, single_gate_time_ns, cz_gate_time_ns.
    """
    cal_id = metrics_data.get("calibration_set_id", "unknown")
    timestamp = metrics_data.get(
        "calibration_set_end_timestamp",
        metrics_data.get("timestamp", datetime.utcnow().isoformat()),
    )
    metrics = metrics_data.get("metrics", {})

    print(f"\n  Raw response top-level keys: {list(metrics_data.keys())}")
    print(f"  Metrics keys: {list(metrics.keys())}")

    # ── Pivot: metric_type -> qubit -> value  =>  qubit -> metric -> value ──
    qubits_raw = {}
    gates_raw = {}

    for metric_name, qubit_values in metrics.items():
        if not isinstance(qubit_values, dict):
            continue
        for qubit_key, entry in qubit_values.items():
            if qubit_key == "statistics":
                continue
            val = entry.get("value") if isinstance(entry, dict) else entry
            if not isinstance(val, (int, float)):
                continue
            # Two-qubit gate: QB1__QB2 (VTT double underscore) -> QB1-QB2
            if "__" in qubit_key:
                normalized = qubit_key.replace("__", "-")
                gates_raw.setdefault(normalized, {})[metric_name] = val
            elif qubit_key.startswith("QB"):
                qubits_raw.setdefault(qubit_key, {})[metric_name] = val

    # ── Map VTT metric names -> HPCQC fields ──
    #
    # VTT metric name                        -> HPCQC field
    # t1_time (seconds)                      -> t1_us (microseconds)
    # t2_time (seconds)                      -> t2_us (microseconds)
    # t2_echo_time (seconds)                 -> t2_echo_us (microseconds)
    # measure_ssro_constant_fidelity         -> readout_fidelity
    # prx_rb_drag_crf_sx_fidelity            -> 1.0 - single_gate_error
    # cz_irb_crf_crf_fidelity                -> cz_fidelity

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

    two_qubit_gates = {}
    for edge, m in sorted(gates_raw.items()):
        cz_fid = m.get("cz_irb_crf_crf_fidelity")
        if cz_fid is not None:
            two_qubit_gates[edge] = {
                "cz_fidelity": cz_fid,
                "cz_error": 1.0 - cz_fid,
            }

    print(f"  Parsed: {len(qubits)} qubits, {len(two_qubit_gates)} CZ gates")

    # ── Extract topology from architecture endpoints ──
    #
    # Priority: dynamic architecture > static architecture > inferred from gates
    #
    # Dynamic: exactly which qubits/gates were active for THIS calibration.
    # Static:  full physical device topology (all possible connections).
    # Inferred: subset of topology that has calibration data (lossy).
    qubit_connectivity = None
    active_qubits = None
    native_operations = None

    if dynamic_architecture is not None:
        active_qubits = dynamic_architecture.get("qubits")
        dyn_gates = dynamic_architecture.get("gates", {})
        cz_info = dyn_gates.get("cz", {})
        default_impl = cz_info.get("default_implementation", "tgss")
        loci = (cz_info.get("implementations", {})
                       .get(default_impl, {})
                       .get("loci", []))
        if loci:
            qubit_connectivity = [list(pair) for pair in loci]
            print(f"  Dynamic arch: {len(active_qubits or [])} active qubits, "
                  f"{len(qubit_connectivity)} CZ loci")

    if qubit_connectivity is None and architecture is not None:
        qa = architecture.get("quantum_architecture", architecture)
        qubit_connectivity = qa.get("qubit_connectivity")
        active_qubits = active_qubits or qa.get("qubits")
        ops = qa.get("operations", {})
        native_operations = list(ops.keys()) if ops else None
        if qubit_connectivity:
            print(f"  Static arch: {len(active_qubits or [])} qubits, "
                  f"{len(qubit_connectivity)} connectivity edges")

    if qubit_connectivity is None:
        qubit_connectivity = [edge.split("-") for edge in two_qubit_gates]
        print(f"  Connectivity inferred from {len(qubit_connectivity)} calibrated gates")

    # ── Assemble HPCQC calibration JSON ──
    result = {
        "calibration_set_id": cal_id,
        "timestamp": timestamp,
        "device": device_name,
        "adapter": "iqm_v2",
        "notes": f"Fetched from VTT QX API on {datetime.utcnow().isoformat()}",
        "qubits": qubits,
        "two_qubit_gates": two_qubit_gates,
        "qubit_connectivity": qubit_connectivity,
        "single_gate_time_ns": 20,
        "cz_gate_time_ns": 100,
    }

    if active_qubits is not None:
        result["active_qubits"] = active_qubits
    if native_operations is not None:
        result["native_operations"] = native_operations

    return result


def _sort_qb(item):
    """Sort QB1, QB2, ..., QB54 numerically (not lexicographically)."""
    name = item[0]
    try:
        return int(name.replace("QB", ""))
    except ValueError:
        return 9999


def print_summary(cal):
    """Print calibration summary with topology and problem qubit flags."""
    import numpy as np

    qubits = cal["qubits"]
    gates = cal["two_qubit_gates"]
    connectivity = cal.get("qubit_connectivity", [])

    t1s = [q["t1_us"] for q in qubits.values() if q.get("t1_us") is not None]
    t2s = [q["t2_us"] for q in qubits.values() if q.get("t2_us") is not None]
    ros = [q["readout_fidelity"] for q in qubits.values() if q.get("readout_fidelity") is not None]
    ges = [q["single_gate_error"] for q in qubits.values() if q.get("single_gate_error") is not None]
    czs = [g["cz_fidelity"] for g in gates.values() if g.get("cz_fidelity") is not None]

    w = 60
    print(f"\n{'=' * w}")
    print(f"  Q50 CALIBRATION SUMMARY")
    print(f"{'=' * w}")
    print(f"  Calibration ID: {cal['calibration_set_id']}")
    print(f"  Timestamp:      {cal['timestamp']}")
    print(f"  Device:         {cal['device']}")
    print(f"  Qubits:         {len(qubits)}")
    print(f"  CZ gates:       {len(gates)}")
    print(f"  Connectivity:   {len(connectivity)} edges")
    if cal.get("active_qubits"):
        print(f"  Active qubits:  {len(cal['active_qubits'])}")
    if cal.get("native_operations"):
        print(f"  Native ops:     {cal['native_operations']}")
    print()

    if t1s:
        print(f"  T1 (us):       min={min(t1s):6.1f}  median={np.median(t1s):6.1f}  max={max(t1s):6.1f}")
    if t2s:
        print(f"  T2 (us):       min={min(t2s):6.1f}  median={np.median(t2s):6.1f}  max={max(t2s):6.1f}")
    if ros:
        print(f"  Readout fid:   min={min(ros):.4f}  median={np.median(ros):.4f}  max={max(ros):.4f}")
    if ges:
        print(f"  1q gate err:   min={min(ges):.5f}  median={np.median(ges):.5f}  max={max(ges):.5f}")
    if czs:
        print(f"  CZ fidelity:   min={min(czs):.4f}  median={np.median(czs):.4f}  max={max(czs):.4f}")

    # ── Topology completeness check ──
    gate_edges = set(gates.keys())
    conn_edges = {f"{p[0]}-{p[1]}" for p in connectivity}
    if conn_edges and gate_edges:
        uncalibrated = conn_edges - gate_edges
        if uncalibrated:
            print(f"\n  WARNING: {len(uncalibrated)} edges in topology but no calibration data")
        orphan = gate_edges - conn_edges
        if orphan:
            print(f"\n  WARNING: {len(orphan)} calibrated gates not in topology")

    # ── Problem qubits ──
    bad_ro = [(n, q["readout_fidelity"]) for n, q in qubits.items()
              if q.get("readout_fidelity") is not None and q["readout_fidelity"] < 0.90]
    if bad_ro:
        print(f"\n  Low readout fidelity (<90%):")
        for name, fid in sorted(bad_ro, key=lambda x: x[1]):
            print(f"    {name}: {fid:.4f}")

    bad_t2 = [(n, q["t2_us"]) for n, q in qubits.items()
              if q.get("t2_us") is not None and q["t2_us"] < 2.0]
    if bad_t2:
        print(f"\n  Very short T2 (<2 us):")
        for name, val in sorted(bad_t2, key=lambda x: x[1]):
            print(f"    {name}: {val:.2f} us")

    bad_cz = [(n, g["cz_fidelity"]) for n, g in gates.items()
              if g.get("cz_fidelity") is not None and g["cz_fidelity"] < 0.95]
    if bad_cz:
        print(f"\n  Low CZ fidelity (<95%):")
        for name, fid in sorted(bad_cz, key=lambda x: x[1])[:10]:
            print(f"    {name}: {fid:.4f}")
        if len(bad_cz) > 10:
            print(f"    ... and {len(bad_cz) - 10} more")

    print(f"{'=' * w}")


def main():
    # ── Connect to Q50 ──
    DEVICE_CORTEX_URL = os.getenv("Q50_CORTEX_URL")
    if not DEVICE_CORTEX_URL:
        raise EnvironmentError(
            "Q50_CORTEX_URL not set -- are you on q_fiqci partition?"
        )

    provider = IQMProvider(DEVICE_CORTEX_URL, quantum_computer="q50")
    backend = provider.get_backend()
    print(f"Connected to Q50 ({backend.num_qubits} qubits)")

    # ── Set up QX API client ──
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ))
    from lumi_hpc_qc.backends.qx_client import QXClient
    qx = QXClient.from_backend(backend)

    # Optional: specific calibration set from command line
    cal_set_id = sys.argv[1] if len(sys.argv) > 1 else None
    date_str = datetime.utcnow().strftime("%Y%m%d")
    os.makedirs("results", exist_ok=True)
    os.makedirs("examples", exist_ok=True)

    # ── 1. Calibration metrics (critical path — do this first) ──
    print(f"\n-- Fetching calibration metrics --")
    metrics = qx.get_calibration_metrics(cal_set_id)
    if metrics is None:
        raise RuntimeError("Failed to fetch calibration metrics from QX API")

    cal_id = metrics.get("calibration_set_id", "unknown")
    cal_id_short = cal_id[:8] if cal_id != "unknown" else "unknown"

    raw_path = f"results/q50_calibration_raw_{date_str}_{cal_id_short}.json"
    with open(raw_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"  Saved: {raw_path}")

    # ── 2. Static quantum architecture ──
    print(f"\n-- Fetching static quantum architecture --")
    architecture = qx.get_quantum_architecture()
    if architecture:
        arch_path = f"results/q50_architecture_{date_str}.json"
        with open(arch_path, "w") as f:
            json.dump(architecture, f, indent=2, default=str)
        print(f"  Saved: {arch_path}")
    else:
        print(f"  WARNING: Static architecture unavailable")

    # ── 3. Dynamic quantum architecture (per calibration set) ──
    dynamic = None
    if cal_id != "unknown":
        print(f"\n-- Fetching dynamic architecture (cal {cal_id_short}...) --")
        dynamic = qx.get_dynamic_architecture(cal_id)
        if dynamic:
            dyn_path = f"results/q50_dynamic_arch_{date_str}_{cal_id_short}.json"
            with open(dyn_path, "w") as f:
                json.dump(dynamic, f, indent=2, default=str)
            print(f"  Saved: {dyn_path}")
        else:
            print(f"  WARNING: Dynamic architecture unavailable")

    # ── 4. Convert to HPCQC format (metrics + architecture combined) ──
    print(f"\n-- Converting to HPCQC format --")
    hpcqc_cal = convert_to_hpcqc_format(
        metrics,
        architecture=architecture,
        dynamic_architecture=dynamic,
    )

    hpcqc_path = f"examples/q50_calibration_{date_str}_{cal_id_short}.json"
    with open(hpcqc_path, "w") as f:
        json.dump(hpcqc_cal, f, indent=2)
    print(f"  Saved: {hpcqc_path}")

    # ── 5. Summary ──
    print_summary(hpcqc_cal)

    # ── 6. Recent calibration runs ──
    print(f"\n-- Recent calibration runs --")
    runs = qx.get_calibration_runs(limit=5)
    if runs and "results" in runs:
        for run in runs["results"]:
            ok = "PASS" if run.get("successful") else "FAIL"
            print(f"  [{ok}] {run.get('creation_date', '?')[:19]} "
                  f"-> {run.get('status', '?')} "
                  f"(cal: {str(run.get('calibration_set_id', ''))[:8]}...)")

    # ── 7. Backend info (from iqm-client, for cross-reference) ──
    try:
        edges = backend.coupling_map.get_edges()
        print(f"\n  Backend coupling map: {len(edges)} edges")
    except Exception:
        print(f"\n  Backend coupling map: available (could not count edges)")
    print(f"  Native ops: {backend.operation_names}")

    # ── 8. Device health summary (non-critical — last, after all data saved) ──
    qx.print_device_summary()


if __name__ == "__main__":
    main()
