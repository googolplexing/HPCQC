#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Query VTT QX API for granular QPU timing breakdown of completed jobs.

Fetches job timeline, payload (execution parameters), and all available
artifacts for each job ID. Parses phase durations and saves structured
JSON output compatible with sweep_timing.json format.

Uses QXClient to test the full chain:
    IQMProvider -> QXClient.from_backend() -> QX API -> _parse_timeline()

No circuit submission. No QPU cost. Read-only GET requests.

Usage:
    sbatch tests/qpu_timing_query.sh

    # Or query specific job IDs:
    sbatch tests/qpu_timing_query.sh <job_id_1> <job_id_2> ...

Outputs:
    results/qpu_timing_query.json — per-job timing + payload + artifacts
"""
import os
import sys
import json
import time

from iqm.qiskit_iqm import IQMProvider

# ── Known job IDs from timing benchmark (April 7, 2026) ──
DEFAULT_JOBS = [
    ("Batch A: 10x SU2(r=2, 4q)",   "6d90b88e-ec68-4a00-9afb-b81e278c4140"),
    ("Batch B: 1x SU2(r=2, 4q)",    "3d8916ca-0edf-4dae-9fe8-cec93fe700db"),
    ("Batch C: 1x SU2(r=1, 4q)",    "b9fb7b9c-02b8-4c1f-ae0c-455fe4922e75"),
    ("Batch D: 1x Bell(2q)",         "7a9bcbb6-4c0b-481f-b5ec-ff5d04ed8eec"),
    ("Batch E: 10x H-all(53q)",      "7c6a1297-e129-46d4-9728-3216f808a1d9"),
]

# ── QPU execution parameters of interest ──
# These affect per-shot timing and should be captured for benchmarking
QPU_EXEC_PARAMS = [
    "shots",
    "active_reset_cycles",
    "heralding_mode",
    "dd_mode",
    "dd_strategy",
    "max_circuit_duration_over_t2",
    "move_gate_validation",
    "move_gate_frame_tracking",
    "move_validation_mode",
    "move_gate_frame_tracking_mode",
]

# ── All artifact types to probe ──
ARTIFACT_TYPES = [
    "measurements",
    "measurement_counts",
    "runtime_estimates",
    "sweep_results",
]


def main():
    # ── Connect ──
    DEVICE_CORTEX_URL = os.getenv("Q50_CORTEX_URL")
    if not DEVICE_CORTEX_URL:
        raise EnvironmentError(
            "Q50_CORTEX_URL not set -- are you on q_fiqci partition?"
        )

    provider = IQMProvider(DEVICE_CORTEX_URL, quantum_computer="q50")
    backend = provider.get_backend()
    print(f"Connected to Q50 ({backend.num_qubits} qubits)")

    # ── Set up QXClient ──
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"
    ))
    from lumi_hpc_qc.backends.qx_client import QXClient, QPUJobTiming
    qx = QXClient.from_backend(backend)

    # ── Determine job IDs ──
    if len(sys.argv) > 1:
        jobs = [(f"CLI arg {i+1}", jid) for i, jid in enumerate(sys.argv[1:])]
    else:
        jobs = DEFAULT_JOBS

    # ── Device context at query time ──
    print(f"\n-- Device context --")
    queue_len = qx.get_queue_length()
    print(f"  Queue length: {queue_len}")

    device_health = qx.get_device_health()
    health_status = None
    if device_health:
        devices = device_health if isinstance(device_health, list) else [device_health]
        for dev in devices:
            if str(dev.get("id", "")).lower() == "q50":
                health_status = dev.get("health")
                print(f"  Health: {health_status}")
                for svc in dev.get("services", []):
                    print(f"    {svc.get('name')}: {svc.get('status')}")

    policy = qx.get_job_policy()
    if policy:
        print(f"  Batch limit: {policy.get('max_number_circuits_per_batch')}")
        print(f"  Max shots: {policy.get('max_number_shots_per_job')}")

    print(f"\n{'=' * 70}")
    print(f"  QPU TIMING QUERY — {len(jobs)} jobs")
    print(f"{'=' * 70}")

    all_results = []

    for label, job_id in jobs:
        print(f"\n{'─' * 70}")
        print(f"  {label}: {job_id}")
        print(f"{'─' * 70}")

        result_entry = {
            "label": label,
            "job_id": job_id,
        }

        # ── 1. Job status + timeline ──
        status_data = qx.get_job_status(job_id)
        if status_data is None:
            print(f"  FAILED: could not fetch job status")
            result_entry["error"] = "status fetch failed"
            all_results.append(result_entry)
            continue

        print(f"  Status: {status_data.get('status', 'unknown')}")

        # ── 2. Parse timeline ──
        timeline = status_data.get("timeline", [])
        print(f"  Timeline: {len(timeline)} entries")

        if timeline:
            print(f"  Phase timestamps:")
            for entry in timeline:
                if isinstance(entry, dict):
                    s = entry.get("status", entry.get("state", "?"))
                    t = entry.get("timestamp", entry.get("time", "?"))
                    print(f"    {s}: {t}")

        # Parse with QXClient
        timing = QPUJobTiming(job_id=job_id)
        timing.timeline_raw = timeline
        qx._parse_timeline(timing, timeline)

        print(f"\n  Parsed durations:")
        print(f"    QX overhead  (Created→Pending Exec):   {_fmt(timing.qx_overhead_s)}")
        print(f"    Compilation  (Comp Started→Ended):     {_fmt(timing.qx_compile_s)}")
        print(f"    QPU queue    (Pending Exec→Exec Start):{_fmt(timing.qpu_queue_s)}")
        print(f"    QPU execute  (Exec Started→Ended):     {_fmt(timing.qpu_execute_s)}")
        print(f"    Postprocess  (Exec Ended→Ready):       {_fmt(timing.qx_postprocess_s)}")
        print(f"    Delivery     (Ready→Completed):        {_fmt(timing.qx_delivery_s)}")

        result_entry["status"] = status_data.get("status")
        result_entry["timing"] = QXClient.timing_to_dict(timing)
        result_entry["timeline_entry_count"] = len(timeline)
        result_entry["timeline_raw"] = timeline

        # ── 3. Job payload (execution parameters) ──
        payload = qx.get_job_payload(job_id)
        if payload is not None and isinstance(payload, dict):
            # Extract execution parameters
            exec_params = {}
            for key in QPU_EXEC_PARAMS:
                if key in payload:
                    exec_params[key] = payload[key]

            n_circuits = len(payload.get("circuits", [])) if isinstance(payload.get("circuits"), list) else "?"
            cal_set_id = payload.get("calibration_set_id")
            qubit_mapping = payload.get("qubit_mapping")

            print(f"\n  Payload:")
            print(f"    Circuits: {n_circuits}")
            print(f"    Calibration set: {cal_set_id}")
            if qubit_mapping:
                # Flatten qubit mapping for display
                if isinstance(qubit_mapping, list):
                    flat = []
                    for entry in qubit_mapping[:3]:
                        if isinstance(entry, dict):
                            flat.append(str(entry))
                        else:
                            flat.append(str(entry))
                    more = f" ... +{len(qubit_mapping)-3}" if len(qubit_mapping) > 3 else ""
                    print(f"    Qubit mapping: {flat}{more}")
                else:
                    mapping_str = str(qubit_mapping)
                    if len(mapping_str) > 200:
                        mapping_str = mapping_str[:200] + "..."
                    print(f"    Qubit mapping: {mapping_str}")

            print(f"    Execution parameters:")
            for key, val in exec_params.items():
                print(f"      {key}: {val}")

            result_entry["payload"] = {
                "n_circuits": n_circuits,
                "calibration_set_id": cal_set_id,
                "qubit_mapping": qubit_mapping,
                "execution_parameters": exec_params,
                "all_keys": list(payload.keys()),
            }

            # Per-circuit timing (if we have QPU execute time and circuit count)
            if timing.qpu_execute_s is not None and isinstance(n_circuits, int) and n_circuits > 0:
                per_circuit = timing.qpu_execute_s / n_circuits
                shots = exec_params.get("shots", 4096)
                per_shot = (per_circuit / shots * 1e6) if shots else None
                print(f"\n  Derived:")
                print(f"    Per circuit: {per_circuit:.3f}s")
                if per_shot is not None:
                    print(f"    Per shot:    {per_shot:.1f} us")
                result_entry["derived"] = {
                    "per_circuit_s": round(per_circuit, 6),
                    "per_shot_us": round(per_shot, 1) if per_shot else None,
                    "shots": shots,
                }
        else:
            print(f"\n  Payload: not available")

        # ── 4. Probe all artifact types ──
        print(f"\n  Artifacts:")
        artifacts_available = {}
        for artifact_type in ARTIFACT_TYPES:
            art = qx.get_job_artifact(job_id, artifact_type)
            if art is not None:
                # Check for "not available" responses disguised as 200
                is_real = True
                if isinstance(art, dict) and "detail" in art:
                    detail = art["detail"]
                    if isinstance(detail, list) and any("not available" in str(d).lower() for d in detail):
                        is_real = False
                    elif isinstance(detail, str) and "not available" in detail.lower():
                        is_real = False

                if is_real:
                    # Summarize without dumping full data
                    if isinstance(art, dict):
                        summary = f"keys={list(art.keys())[:5]}"
                        if "measurements" in art:
                            summary += f", entries={len(art['measurements']) if isinstance(art['measurements'], (list,dict)) else '?'}"
                    else:
                        summary = f"type={type(art).__name__}"
                    print(f"    {artifact_type}: AVAILABLE ({summary})")
                    artifacts_available[artifact_type] = True
                else:
                    print(f"    {artifact_type}: not available (200 but empty)")
                    artifacts_available[artifact_type] = False
            else:
                print(f"    {artifact_type}: not available")
                artifacts_available[artifact_type] = False

        result_entry["artifacts_available"] = artifacts_available

        # ── 5. Frontend job results (different endpoint, may have extra metadata) ──
        frontend = qx.get_job_results(job_id)
        if frontend is not None and isinstance(frontend, dict):
            frontend_keys = list(frontend.keys())
            # Don't save full measurement data — just metadata
            frontend_meta = {k: v for k, v in frontend.items()
                           if k not in ("measurements", "counts", "results")}
            if len(str(frontend_meta)) > 2000:
                frontend_meta = {"keys": frontend_keys, "truncated": True}
            print(f"\n  Frontend results endpoint:")
            print(f"    Keys: {frontend_keys}")
            result_entry["frontend_metadata"] = frontend_meta
        else:
            print(f"\n  Frontend results: not available")

        all_results.append(result_entry)

    # ── Summary tables ──
    print(f"\n{'=' * 70}")
    print(f"  TIMING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Label':<30} {'QPU Queue':>10} {'QPU Exec':>10} {'Compile':>10} {'Per Circ':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['label']:<30} {'FAILED':>40}")
            continue
        t = r["timing"]["qx_server"]
        derived = r.get("derived", {})
        per_circ = derived.get("per_circuit_s")
        print(f"  {r['label']:<30} "
              f"{_fmt(t.get('qpu_queue_s')):>10} "
              f"{_fmt(t.get('qpu_execute_s')):>10} "
              f"{_fmt(t.get('compile_s')):>10} "
              f"{_fmt(per_circ):>10}")

    # ── Execution parameters comparison ──
    print(f"\n{'=' * 70}")
    print(f"  EXECUTION PARAMETERS")
    print(f"{'=' * 70}")
    for r in all_results:
        if "error" in r or "payload" not in r:
            continue
        ep = r["payload"].get("execution_parameters", {})
        print(f"\n  {r['label']}:")
        for key in QPU_EXEC_PARAMS:
            if key in ep:
                print(f"    {key}: {ep[key]}")

    # ── Artifact availability matrix ──
    print(f"\n{'=' * 70}")
    print(f"  ARTIFACT AVAILABILITY")
    print(f"{'=' * 70}")
    print(f"  {'Label':<30} {'measures':>10} {'counts':>10} {'runtime':>10} {'sweep':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for r in all_results:
        if "error" in r:
            continue
        aa = r.get("artifacts_available", {})
        print(f"  {r['label']:<30} "
              f"{'YES' if aa.get('measurements') else 'no':>10} "
              f"{'YES' if aa.get('measurement_counts') else 'no':>10} "
              f"{'YES' if aa.get('runtime_estimates') else 'no':>10} "
              f"{'YES' if aa.get('sweep_results') else 'no':>10}")

    print(f"\n{'=' * 70}")

    # ── Save results ──
    os.makedirs("results", exist_ok=True)
    out_path = "results/qpu_timing_query.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": "Q50",
            "device_context": {
                "queue_length": queue_len,
                "health": health_status,
                "job_policy": policy,
            },
            "jobs_queried": len(all_results),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


def _fmt(val):
    """Format a timing value for display."""
    if val is None:
        return "-"
    if val < 0.01:
        return f"{val*1000:.1f}ms"
    return f"{val:.3f}s"


if __name__ == "__main__":
    main()
