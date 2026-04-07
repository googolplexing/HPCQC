#!/usr/bin/env python3
"""Inspect completed Q50 job metadata for QPU execution timing.

Queries the IQM Cortex API for a completed job ID and prints all
available attributes — looking for execution duration, timestamps,
or any timing data that separates actual QPU time from queue/network.

Usage:
    sbatch tests/qpu_job_inspect.sh <job_id>

    # Or inspect all 4 completed jobs from the timing benchmark:
    sbatch tests/qpu_job_inspect.sh

If no job_id is provided, inspects all job IDs from the timing run.
"""
import os
import sys
import json
from datetime import datetime

from iqm.qiskit_iqm import IQMProvider

# ── Connect to Q50 ──
DEVICE_CORTEX_URL = os.getenv("Q50_CORTEX_URL")
if not DEVICE_CORTEX_URL:
    raise EnvironmentError("Q50_CORTEX_URL not set — are you on q_fiqci partition?")

provider = IQMProvider(DEVICE_CORTEX_URL, quantum_computer="q50")
backend = provider.get_backend()
client = backend.client

# Job IDs from the timing benchmark (Batches A-D that completed)
DEFAULT_JOB_IDS = [
    ("Batch A (10x4q)", "6d90b88e-ec68-4a00-9afb-b81e278c4140"),
    ("Batch B (1x4q)",  "3d8916ca-0edf-4dae-9fe8-cec93fe700db"),
    ("Batch C (1x4q shallow)", "b9fb7b9c-02b8-4c1f-ae0c-455fe4922e75"),
    ("Batch D (1x2q Bell)",    "7a9bcbb6-4c0b-481f-b5ec-ff5d04ed8eec"),
]

# Use command-line arg if provided, otherwise inspect all defaults
if len(sys.argv) > 1:
    job_ids = [("CLI arg", sys.argv[1])]
else:
    job_ids = DEFAULT_JOB_IDS

print("═══════════════════════════════════════════════════════")
print("  Q50 JOB METADATA INSPECTION")
print("═══════════════════════════════════════════════════════")
print(f"  Jobs to inspect: {len(job_ids)}")
print()

# ── Inspect IQM client internals for timing-related methods ──
print("── IQM client API surface ──")
print(f"  Client type: {type(client).__name__}")

# Check client methods
client_methods = [m for m in dir(client) if not m.startswith('_')]
print(f"  Client methods ({len(client_methods)}):")
for m in sorted(client_methods):
    print(f"    {m}")
print()

# Check server client methods
sc = client._iqm_server_client
sc_methods = [m for m in dir(sc) if not m.startswith('_')]
timing_methods = [m for m in sc_methods
                  if any(kw in m.lower() for kw in
                         ['time', 'duration', 'metric', 'status', 'job', 'result'])]
print(f"  Server client timing-related methods:")
for m in sorted(timing_methods):
    print(f"    {m}")
print()

# ── Inspect each job ──
all_results = []

for label, job_id in job_ids:
    print(f"═══════════════════════════════════════════════════════")
    print(f"  {label}: {job_id}")
    print(f"═══════════════════════════════════════════════════════")

    job_data = {}

    # Method 1: client.get_job() — standard Qiskit IQM path
    try:
        job = client.get_job(job_id)
        job_type = type(job).__name__
        print(f"\n  client.get_job() → {job_type}")

        attrs = [a for a in dir(job) if not a.startswith('_')]
        print(f"  Attributes ({len(attrs)}):")
        for attr in sorted(attrs):
            try:
                val = getattr(job, attr)
                if not callable(val):
                    val_str = str(val)
                    if len(val_str) > 200:
                        val_str = val_str[:200] + "..."
                    print(f"    {attr}: {val_str}")
                    job_data[attr] = val_str
            except Exception as e:
                print(f"    {attr}: <error: {e}>")
    except Exception as e:
        print(f"\n  client.get_job() failed: {e}")

    # Method 2: server client get_job_status
    try:
        status = sc.get_job_status(job_id)
        print(f"\n  server_client.get_job_status() → {type(status).__name__}")
        if hasattr(status, '__dict__'):
            for k, v in status.__dict__.items():
                print(f"    {k}: {v}")
                job_data[f"status_{k}"] = str(v)
        else:
            print(f"    value: {status}")
            job_data["status"] = str(status)
    except Exception as e:
        print(f"\n  server_client.get_job_status() failed: {e}")

    # Method 3: Try raw HTTP to the job endpoint
    try:
        import requests
        headers = {"User-Agent": sc._signature}
        bearer = sc._auth_header_callback()
        headers["Authorization"] = bearer
        url = f"{sc.root_url}/api/jobs/{job_id}"
        resp = requests.get(url, headers=headers, timeout=10)
        if resp.status_code == 200:
            raw = resp.json()
            print(f"\n  Raw API /api/jobs/{{id}} response keys: {list(raw.keys())}")
            for k, v in raw.items():
                v_str = str(v)
                if len(v_str) > 300:
                    v_str = v_str[:300] + "..."
                print(f"    {k}: {v_str}")
                job_data[f"raw_{k}"] = v_str
        else:
            print(f"\n  Raw API returned {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"\n  Raw API request failed: {e}")

    # Method 4: Try /api/jobs/{id}/metrics or /api/jobs/{id}/timing
    for endpoint in ["metrics", "timing", "details", "metadata"]:
        try:
            url = f"{sc.root_url}/api/jobs/{job_id}/{endpoint}"
            resp = requests.get(url, headers=headers, timeout=10)
            if resp.status_code == 200:
                print(f"\n  /api/jobs/{{id}}/{endpoint}: {resp.json()}")
                job_data[f"endpoint_{endpoint}"] = resp.text[:500]
            elif resp.status_code != 404:
                print(f"\n  /api/jobs/{{id}}/{endpoint}: HTTP {resp.status_code}")
        except Exception:
            pass

    all_results.append({"label": label, "job_id": job_id, "data": job_data})
    print()

# ── Save results ──
out_path = "results/qpu_job_inspect.json"
os.makedirs("results", exist_ok=True)
with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, default=str)
print(f"Results saved: {out_path}")

# ── Summary: which timing fields did we find? ──
print()
print("═══════════════════════════════════════════════════════")
print("  TIMING FIELDS FOUND")
print("═══════════════════════════════════════════════════════")
timing_keywords = ['time', 'duration', 'start', 'end', 'elapsed',
                   'queue', 'execute', 'compile', 'submit', 'complete']
for result in all_results:
    print(f"\n  {result['label']}:")
    found_any = False
    for key, val in result['data'].items():
        if any(kw in key.lower() for kw in timing_keywords):
            print(f"    {key}: {val}")
            found_any = True
    if not found_any:
        print(f"    (no timing fields found)")
print()
print("═══════════════════════════════════════════════════════")
