#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Query VTT QX API for granular QPU timing breakdown of completed jobs.

Fetches job timeline and runtime_estimates artifacts for each job ID,
parses the phase durations, and saves structured JSON output compatible
with sweep_timing.json format.

Uses QXClient to test the full chain:
    IQMProvider → QXClient.from_backend() → QX API → _parse_timeline()

No circuit submission. No QPU cost. Read-only GET requests.

Usage:
    sbatch tests/qpu_timing_query.sh

    # Or query specific job IDs:
    sbatch tests/qpu_timing_query.sh <job_id_1> <job_id_2> ...

Outputs:
    results/qpu_timing_query.json — per-job timing breakdown
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

    print(f"\n{'=' * 70}")
    print(f"  QPU TIMING QUERY — {len(jobs)} jobs")
    print(f"{'=' * 70}")

    all_results = []

    for label, job_id in jobs:
        print(f"\n-- {label}: {job_id[:12]}... --")

        # ── 1. Fetch job status + timeline ──
        status_data = qx.get_job_status(job_id)
        if status_data is None:
            print(f"  FAILED: could not fetch job status")
            all_results.append({
                "label": label, "job_id": job_id, "error": "status fetch failed"
            })
            continue

        print(f"  Status: {status_data.get('status', 'unknown')}")
        print(f"  Type: {status_data.get('job_type', 'unknown')}")

        # ── 2. Parse timeline ──
        timeline = status_data.get("timeline", [])
        print(f"  Timeline entries: {len(timeline)}")

        # Print raw timeline
        if timeline:
            print(f"  Raw timeline:")
            for entry in timeline:
                if isinstance(entry, dict):
                    s = entry.get("status", entry.get("state", "?"))
                    t = entry.get("timestamp", entry.get("time", "?"))
                    print(f"    {s}: {t}")

        # Use QXClient's parser
        timing = QPUJobTiming(job_id=job_id)
        timing.timeline_raw = timeline
        qx._parse_timeline(timing, timeline)

        print(f"\n  Parsed timing:")
        print(f"    QX overhead (Created→Pending Execution):  {_fmt(timing.qx_overhead_s)}")
        print(f"    QX compile (Compilation Started→Ended):   {_fmt(timing.qx_compile_s)}")
        print(f"    QPU queue (Pending Execution→Exec Start): {_fmt(timing.qpu_queue_s)}")
        print(f"    QPU execute (Exec Started→Ended):         {_fmt(timing.qpu_execute_s)}")
        print(f"    QX postprocess (Exec Ended→Ready):        {_fmt(timing.qx_postprocess_s)}")
        print(f"    QX delivery (Ready→Completed):            {_fmt(timing.qx_delivery_s)}")

        # ── 3. Fetch runtime_estimates artifact ──
        print(f"\n  Runtime estimates artifact:")
        rt = qx.get_runtime_estimates(job_id)
        if rt is not None:
            print(f"    Keys: {list(rt.keys()) if isinstance(rt, dict) else type(rt).__name__}")
            print(f"    Raw: {json.dumps(rt, indent=6, default=str)[:500]}")
            timing.runtime_estimates_raw = rt
        else:
            print(f"    Not available (endpoint returned None)")

        # ── 4. Also fetch measurement_counts artifact to check availability ──
        mc = qx.get_job_artifact(job_id, "measurement_counts")
        if mc is not None:
            print(f"\n  Measurement counts artifact: available")
            if isinstance(mc, dict):
                print(f"    Keys: {list(mc.keys())[:5]}...")
        else:
            print(f"\n  Measurement counts artifact: not available")

        # ── 5. Fetch job payload to see circuit details ──
        payload = qx.get_job_payload(job_id)
        if payload is not None:
            print(f"\n  Job payload:")
            if isinstance(payload, dict):
                print(f"    Keys: {list(payload.keys())}")
                shots = payload.get("shots", "?")
                circuits = payload.get("circuits", [])
                print(f"    Shots: {shots}")
                print(f"    Circuits: {len(circuits) if isinstance(circuits, list) else '?'}")
        else:
            print(f"\n  Job payload: not available")

        # ── 6. Build result dict (sweep_timing.json compatible) ──
        result = {
            "label": label,
            "job_id": job_id,
            "status": status_data.get("status"),
            "timing": QXClient.timing_to_dict(timing),
            "timeline_entry_count": len(timeline),
            "runtime_estimates_available": rt is not None,
            "runtime_estimates_raw": rt,
        }
        all_results.append(result)

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print(f"  TIMING SUMMARY")
    print(f"{'=' * 70}")
    print(f"  {'Label':<30} {'QPU Queue':>10} {'QPU Exec':>10} {'Compile':>10} {'Total':>10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")

    for r in all_results:
        if "error" in r:
            print(f"  {r['label']:<30} {'FAILED':>10}")
            continue
        t = r["timing"]["qx_server"]
        total = sum(v for v in [
            t.get("overhead_s"), t.get("qpu_queue_s"),
            t.get("qpu_execute_s"), t.get("postprocess_s"),
            t.get("delivery_s"),
        ] if v is not None)
        print(f"  {r['label']:<30} "
              f"{_fmt(t.get('qpu_queue_s')):>10} "
              f"{_fmt(t.get('qpu_execute_s')):>10} "
              f"{_fmt(t.get('compile_s')):>10} "
              f"{_fmt(total):>10}")

    print(f"{'=' * 70}")

    # ── Save results ──
    os.makedirs("results", exist_ok=True)
    out_path = "results/qpu_timing_query.json"
    with open(out_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "device": "Q50",
            "jobs_queried": len(all_results),
            "results": all_results,
        }, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}")


def _fmt(val):
    """Format a timing value for display."""
    if val is None:
        return "—"
    if val < 0.01:
        return f"{val*1000:.1f}ms"
    return f"{val:.3f}s"


if __name__ == "__main__":
    main()
