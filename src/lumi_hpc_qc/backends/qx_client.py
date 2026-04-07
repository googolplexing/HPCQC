# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""VTT QX API client — device architecture, job timing, and monitoring.

Augments iqm-client with QX-specific REST endpoints that iqm-client
does not expose: quantum architecture, job timeline, runtime estimates,
queue monitoring, and QPU usage tracking.

AUTHENTICATION CONSTRAINT (CRITICAL):
    The IQM bearer token is NEVER known to this client. On LUMI, the
    FiQCI middleware inserts the real IQM credential via a proxy layer
    that intercepts outbound requests. Our code receives an opaque auth
    header from iqm-client's internal callback — we pass it through
    without inspecting, storing, or assuming anything about its contents.

    ALL requests MUST route through the same root_url that iqm-client
    uses (the proxy endpoint), never directly to qx.vtt.fi. The proxy
    decides which endpoints to forward and which credentials to inject.

    If the proxy does not forward a particular QX API path, the request
    will fail. All methods return None on failure for this reason.

FRAGILITY NOTE:
    from_backend() accesses iqm-client internals:
        backend.client._iqm_server_client._signature
        backend.client._iqm_server_client._auth_header_callback()
        backend.client._iqm_server_client.root_url
        backend.client._iqm_server_client.quantum_computer
    These are private attributes. If IQM changes their client library
    internals, this will break. There is no public API for extracting
    auth headers from an IQMBackend. This is the same pattern used by
    CSC's own FiQCI examples (get_calibration_data.py).

VTT QX API version: qx-v0.6.7
Base URL: routed through FiQCI proxy (never hardcoded)
API schema reference: VTT_QX_-_Quantum_Computing_API.yaml
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import requests


# ═══════════════════════════════════════════════════════════════════════
# Timing Data Structure
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class QPUJobTiming:
    """Complete timing breakdown for a single QPU batch submission.

    Captures all 5 phases of QPU job execution:

        Phase 1 — SLURM queue:    Waiting for q_fiqci partition.
                                  Captured from SLURM env vars externally.
        Phase 2 — VTT QX queue:   Waiting in VTT's job queue.
                                  From job timeline: submitted → compilation.
        Phase 3 — QX compilation: Server-side transpilation/compilation.
                                  From job timeline: compilation → execution.
        Phase 4 — QPU execution:  Actual quantum processor time.
                                  From runtime_estimates artifact.
        Phase 5 — Retrieval:      Result transfer back to LUMI.
                                  Local perf_counter around .result() call.

    Local fields (wall_total_s, local_submit_s, local_wait_s) are always
    populated. QX server-side fields are populated when the API returns
    data, None otherwise. This ensures timing capture never blocks or
    crashes the main execution path.
    """
    job_id: str
    n_circuits: int = 0
    shots: int = 0

    # ── Local timing (time.perf_counter on LUMI) ──
    wall_total_s: float = 0.0       # .run() call → .result() return
    local_submit_s: float = 0.0     # .run() call duration (before block)
    local_wait_s: float = 0.0       # .result() blocking duration

    # ── QX server-side timing (from job timeline / artifacts) ──
    qx_queue_s: float | None = None       # submitted → compilation start
    qx_compile_s: float | None = None     # compilation → execution start
    qx_execute_s: float | None = None     # execution start → end
    qx_retrieve_s: float | None = None    # execution end → result ready
    qpu_seconds: float | None = None      # billed QPU seconds (usage API)

    # ── Context at submission time ──
    queue_length_before: int | None = None

    # ── Raw API responses (for provenance / debugging) ──
    timeline_raw: list | None = None
    runtime_estimates_raw: dict | None = None
    job_status_raw: dict | None = None


# ═══════════════════════════════════════════════════════════════════════
# QX REST Client
# ═══════════════════════════════════════════════════════════════════════

class QXClient:
    """Thin REST client for VTT QX API endpoints.

    Does NOT replace iqm-client for circuit submission. Provides access
    to QX endpoints that iqm-client does not expose: architecture,
    timing, monitoring, usage.

    Every method returns None on failure (network errors, auth issues,
    proxy doesn't forward the endpoint, device offline). Callers must
    handle None gracefully — this is by design, not a bug.
    """

    def __init__(
        self,
        root_url: str,
        device: str,
        auth_headers: dict[str, str],
    ) -> None:
        """Direct construction. Prefer QXClient.from_backend().

        Args:
            root_url: Proxy endpoint URL (NOT qx.vtt.fi directly).
            device: Device slug (e.g. "q50").
            auth_headers: Opaque headers from iqm-client. Passed through
                          as-is — contents are proxy-managed.
        """
        self._root_url = root_url.rstrip("/")
        self._device = device
        self._headers = dict(auth_headers)
        self._timeout = 30

    @classmethod
    def from_backend(cls, backend) -> QXClient:
        """Create QXClient from an existing IQM backend connection.

        Extracts the proxy URL and opaque auth headers from iqm-client
        internals. The bearer token is proxy-managed — we never see the
        real IQM credential.

        Args:
            backend: IQMBackend from IQMProvider.get_backend().

        Raises:
            AttributeError: If iqm-client internals changed (fragile).
        """
        sc = backend.client._iqm_server_client
        headers = {"User-Agent": sc._signature}
        bearer = sc._auth_header_callback()
        headers["Authorization"] = bearer
        return cls(
            root_url=sc.root_url,
            device=sc.quantum_computer,
            auth_headers=headers,
        )

    def _get(self, path: str) -> dict | None:
        """GET request through the proxy. Returns None on any failure."""
        url = f"{self._root_url}{path}"
        try:
            resp = requests.get(
                url, headers=self._headers, timeout=self._timeout,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f"  QX API [{path}]: {e}")
            return None

    # ───────────────────────────────────────────────────────────────
    # Device Architecture
    # ───────────────────────────────────────────────────────────────

    def get_quantum_architecture(self) -> dict | None:
        """Static device topology: qubits, connectivity, native operations.

        GET /api/devices/{device}/quantum-architecture

        Returns the physical device structure including qubit_connectivity
        (list of [QB1, QB2] pairs) which is the authoritative topology
        source. This does not change between calibration cycles.
        """
        return self._get(
            f"/api/devices/{self._device}/quantum-architecture"
        )

    def get_dynamic_architecture(self, calibration_set_id: str) -> dict | None:
        """Per-calibration active qubits and gate loci.

        GET /api/v1/calibration-sets/{device}/{cal_set_id}/dynamic-quantum-architecture

        Reflects which qubits/gates were active during a specific
        calibration cycle. Deactivated qubits (e.g. QB32 on Q50) are
        excluded, which changes the qubit index mapping. This is the
        authoritative source for calibration-set-dependent topology.
        """
        return self._get(
            f"/api/v1/calibration-sets/{self._device}/"
            f"{calibration_set_id}/dynamic-quantum-architecture"
        )

    # ───────────────────────────────────────────────────────────────
    # Calibration Metrics
    # ───────────────────────────────────────────────────────────────

    def get_calibration_metrics(
        self, calibration_set_id: str | None = None,
    ) -> dict | None:
        """Per-qubit/gate calibration metrics (T1, T2, fidelities).

        GET /api/devices/{device}/calibration/metrics/latest
        GET /api/devices/{device}/calibration/metrics/{calibration_set_id}
        """
        if calibration_set_id:
            path = (f"/api/devices/{self._device}"
                    f"/calibration/metrics/{calibration_set_id}")
        else:
            path = f"/api/devices/{self._device}/calibration/metrics/latest"
        return self._get(path)

    def get_calibration_runs(
        self, limit: int = 5, include_metrics: bool = False,
    ) -> dict | None:
        """Recent calibration runs (last 30 days for non-superusers).

        GET /api/calibration/{device}/runs
        """
        params = f"?limit={limit}&order=descending"
        if include_metrics:
            params += "&include_metrics=true"
        return self._get(f"/api/calibration/{self._device}/runs{params}")

    # ───────────────────────────────────────────────────────────────
    # Device Monitoring
    # ───────────────────────────────────────────────────────────────

    def get_device_health(self) -> dict | None:
        """Device health: online/offline status and service states.

        GET /api/devices/health
        """
        return self._get("/api/devices/health")

    def get_queue_length(self) -> int | None:
        """Current job queue depth for this device.

        GET /api/devices/{device}/queue-length
        """
        data = self._get(f"/api/devices/{self._device}/queue-length")
        if data is not None:
            return data.get("queue_length")
        return None

    def get_device_list(self) -> dict | None:
        """All devices with job policies (batch limits, shot limits).

        GET /api/devices/list

        The job_policy field contains max_number_circuits_per_batch,
        which should replace the hardcoded VTT_BATCH_LIMIT = 200 in
        iqm_qpu.py when available.
        """
        return self._get("/api/devices/list")

    def get_job_policy(self) -> dict | None:
        """Extract job policy for this device from device list.

        Returns e.g.:
            {"max_number_circuits_per_batch": 200,
             "max_number_shots_per_job": 100000,
             "max_queue_length": 500}
        """
        devices = self.get_device_list()
        if devices is None:
            return None
        device_list = devices
        if isinstance(devices, dict):
            device_list = devices.get("quantum_computers",
                                      devices.get("results", []))
        if not isinstance(device_list, list):
            return None
        for dev in device_list:
            slug = dev.get("slug", dev.get("id", ""))
            if slug.lower() == self._device.lower():
                return dev.get("job_policy")
        return None

    def is_device_online(self) -> bool:
        """Quick check: is the device accepting jobs?"""
        health = self.get_device_health()
        if health is None:
            return False
        devices = health if isinstance(health, list) else [health]
        for dev in devices:
            if dev.get("id", "").lower() == self._device.lower():
                return dev.get("health", "").lower() == "online"
        return False

    # ───────────────────────────────────────────────────────────────
    # Job Timing and Artifacts
    # ───────────────────────────────────────────────────────────────

    def get_job_status(self, job_id: str) -> dict | None:
        """Job status with timeline of phase transitions.

        GET /api/v1/jobs/{job_id}

        The timeline field contains timestamps for each status change:
        submitted, pending_compilation, pending_execution, executing, ready.
        """
        return self._get(f"/api/v1/jobs/{job_id}")

    def get_job_results(self, job_id: str) -> dict | None:
        """Job results: status + measurements (frontend format).

        GET /api/jobs/{job_id}
        """
        return self._get(f"/api/jobs/{job_id}")

    def get_job_artifact(self, job_id: str, artifact_type: str) -> dict | None:
        """Retrieve a specific job artifact.

        GET /api/v1/jobs/{job_id}/artifacts/{artifact_type}

        Supported artifact types:
            measurements       — raw per-shot bitstrings
            measurement_counts — aggregated counts
            runtime_estimates  — QPU execution timing data
            sweep_results      — raw protobuf (pulse-level jobs)
        """
        return self._get(
            f"/api/v1/jobs/{job_id}/artifacts/{artifact_type}"
        )

    def get_runtime_estimates(self, job_id: str) -> dict | None:
        """QPU execution time estimates for a completed job.

        GET /api/v1/jobs/{job_id}/artifacts/runtime_estimates

        Returns actual QPU execution time, separated from queue wait
        and compilation. This is the number that matters for
        drain_timeout calibration in the cross-seed pool dispatcher.
        """
        return self.get_job_artifact(job_id, "runtime_estimates")

    def get_job_payload(self, job_id: str) -> dict | None:
        """Job payload: circuits and parameters as submitted.

        GET /api/v1/jobs/{job_id}/payload
        """
        return self._get(f"/api/v1/jobs/{job_id}/payload")

    # ───────────────────────────────────────────────────────────────
    # QPU Usage
    # ───────────────────────────────────────────────────────────────
    #
    # The QX API defines GET /api/projects/{project_id}/usage but this
    # endpoint is likely inaccessible through the FiQCI proxy — the
    # injected token is scoped for QPU job submission, not project
    # administration. Allocation tracking on LUMI uses CSC's tool:
    #
    #   /appl/local/quantum/resource-checker/project-qpu-allocations project_462001126
    #
    # If this endpoint becomes accessible in a future FiQCI module
    # version, add get_project_usage(project_id) here.

    # ───────────────────────────────────────────────────────────────
    # Timing Capture
    # ───────────────────────────────────────────────────────────────

    def capture_job_timing(
        self,
        backend,
        circuits,
        shots: int,
    ) -> tuple[Any, QPUJobTiming]:
        """Submit circuits via iqm-client and capture full timing breakdown.

        Wraps backend.run() — circuit submission still goes through
        iqm-client and the FiQCI proxy normally. After results return,
        queries QX API for server-side timing data.

        Args:
            backend: IQM backend (IqmQpuBackend._sim).
            circuits: Single QuantumCircuit or list of circuits.
            shots: Number of measurement shots.

        Returns:
            (qiskit_result, timing_record) tuple.
        """
        is_list = isinstance(circuits, list)
        n = len(circuits) if is_list else 1

        # Pre-submission context
        queue_len = self.get_queue_length()

        # Submit through iqm-client (proxy handles auth)
        t0 = time.perf_counter()
        job = backend.run(circuits, shots=shots)
        t_submitted = time.perf_counter()

        job_id = str(job.job_id())

        # Wait for result (blocks until QPU completes + results transfer)
        result = job.result()
        t_done = time.perf_counter()

        timing = QPUJobTiming(
            job_id=job_id,
            n_circuits=n,
            shots=shots,
            wall_total_s=t_done - t0,
            local_submit_s=t_submitted - t0,
            local_wait_s=t_done - t_submitted,
            queue_length_before=queue_len,
        )

        # Enrich with server-side timing (best-effort, never blocks)
        self._enrich_timing(timing)

        return result, timing

    def _enrich_timing(self, timing: QPUJobTiming) -> None:
        """Fetch QX API data to populate server-side timing fields.

        Best-effort: failures leave fields as None. This method never
        raises — timing capture must not break the execution path.
        """
        try:
            # Job timeline
            status = self.get_job_status(timing.job_id)
            if status is not None:
                timing.job_status_raw = status
                timeline = status.get("timeline", [])
                timing.timeline_raw = timeline
                self._parse_timeline(timing, timeline)

            # Runtime estimates (actual QPU time)
            rt = self.get_runtime_estimates(timing.job_id)
            if rt is not None:
                timing.runtime_estimates_raw = rt
                if isinstance(rt, dict):
                    for key in ("qpu_time_s", "execution_time_s",
                                "qpu_seconds", "total_time", "duration"):
                        if key in rt:
                            timing.qpu_seconds = float(rt[key])
                            break
        except Exception:
            pass  # timing enrichment must never crash

    def _parse_timeline(self, timing: QPUJobTiming, timeline: list) -> None:
        """Extract phase durations from QX job timeline entries.

        Timeline entries are expected to have status/timestamp fields.
        The exact field names depend on VTT's API version — this handles
        known patterns and degrades gracefully on unknown ones.
        """
        if not timeline:
            return

        from datetime import datetime

        timestamps: dict[str, datetime] = {}
        for entry in timeline:
            if not isinstance(entry, dict):
                continue
            status = entry.get("status", entry.get("state", ""))
            ts_str = entry.get("timestamp", entry.get("time", ""))
            if status and ts_str:
                try:
                    ts = datetime.fromisoformat(
                        ts_str.replace("Z", "+00:00")
                    )
                    timestamps[status.lower().replace(" ", "_")] = ts
                except (ValueError, TypeError):
                    pass

        def _delta(start, end):
            s, e = timestamps.get(start), timestamps.get(end)
            return (e - s).total_seconds() if s and e else None

        timing.qx_queue_s = _delta("submitted", "pending_compilation")
        timing.qx_compile_s = _delta("pending_compilation",
                                     "pending_execution")
        timing.qx_execute_s = _delta("pending_execution", "ready") or \
                              _delta("executing", "ready")

    # ───────────────────────────────────────────────────────────────
    # Utilities
    # ───────────────────────────────────────────────────────────────

    def print_device_summary(self) -> None:
        """Print human-readable device status to stdout."""
        print(f"\n  VTT QX Device: {self._device}")

        health = self.get_device_health()
        if health:
            devices = health if isinstance(health, list) else [health]
            for dev in devices:
                if dev.get("id", "").lower() == self._device.lower():
                    print(f"  Health: {dev.get('health', 'unknown')}")
                    for svc in dev.get("services", []):
                        print(f"    Service {svc.get('name')}: "
                              f"{svc.get('status')}")

        queue = self.get_queue_length()
        if queue is not None:
            print(f"  Queue length: {queue}")

        policy = self.get_job_policy()
        if policy:
            print(f"  Max circuits/batch: "
                  f"{policy.get('max_number_circuits_per_batch')}")
            print(f"  Max shots/job: "
                  f"{policy.get('max_number_shots_per_job')}")

        arch = self.get_quantum_architecture()
        if arch:
            qa = arch.get("quantum_architecture", arch)
            qubits = qa.get("qubits", [])
            conn = qa.get("qubit_connectivity", [])
            ops = list(qa.get("operations", {}).keys())
            print(f"  Qubits: {len(qubits)}")
            print(f"  Connectivity: {len(conn)} edges")
            print(f"  Native ops: {ops}")

    @staticmethod
    def timing_to_dict(timing: QPUJobTiming) -> dict:
        """Serialize QPUJobTiming to JSON-safe dict for sweep_timing.json."""
        return {
            "job_id": timing.job_id,
            "n_circuits": timing.n_circuits,
            "shots": timing.shots,
            "local": {
                "wall_total_s": round(timing.wall_total_s, 4),
                "submit_s": round(timing.local_submit_s, 4),
                "wait_s": round(timing.local_wait_s, 4),
            },
            "qx_server": {
                "queue_s": timing.qx_queue_s,
                "compile_s": timing.qx_compile_s,
                "execute_s": timing.qx_execute_s,
                "retrieve_s": timing.qx_retrieve_s,
                "qpu_seconds": timing.qpu_seconds,
            },
            "context": {
                "queue_length_before": timing.queue_length_before,
            },
        }
