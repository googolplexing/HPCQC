# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Benchmark Parquet export — per-batch QPU timing and packing metrics.

Writes `sweep_benchmark.parquet` alongside `sweep.h5` and
`sweep_results.parquet`. One row per batch submission. For QPU sweeps,
timing comes from QPUJobTiming records captured via QXClient. For
simulator sweeps, local perf_counter data is used and QPU-specific
columns are null.

36-column schema per RED-DIRECTIVE-BENCHMARK-PARQUET-v1.0 Appendix A
+ RED-DIRECTIVE-QPU-CONFIG-v1.0 §5 (retry_attempts):
  Identity (6), Batch (8), Server Timing (6), Local Timing (3),
  Derived (2), Context (5), Retry (1), Infrastructure (5)

RED-DIRECTIVE-V130-v1.0 §1 — Benchmark Parquet Export
RED-DIRECTIVE-QPU-CONFIG-v1.0 §5 — retry_attempts column
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _benchmark_schema():
    """Build the 36-column PyArrow schema.

    Lazy import to avoid pyarrow dependency at module load time
    (pyarrow is heavy and not needed for HDF5-only operations).
    """
    import pyarrow as pa

    return pa.schema([
        # ── Section 1: Identity ──
        pa.field("sweep_id", pa.string()),
        pa.field("job_id", pa.string()),                   # nullable (QPU only)
        pa.field("batch_index", pa.int32()),
        pa.field("framework_version", pa.string()),
        pa.field("timestamp_utc", pa.string()),
        pa.field("mode", pa.string()),                     # "qpu" or "simulator"

        # ── Section 2: Batch ──
        pa.field("n_circuits", pa.int32()),
        pa.field("shots", pa.int32()),
        pa.field("total_shots", pa.int64()),
        pa.field("circuit_width", pa.int32()),
        pa.field("n_placements", pa.int32()),
        pa.field("packing_rounds", pa.int32()),
        pa.field("packing_algorithm", pa.string()),
        pa.field("qubit_utilization_mean", pa.float64()),

        # ── Section 3: Server Timing (QPU only, nullable) ──
        pa.field("qx_overhead_s", pa.float64()),
        pa.field("qx_compile_s", pa.float64()),
        pa.field("qpu_queue_s", pa.float64()),
        pa.field("qpu_execute_s", pa.float64()),
        pa.field("qx_postprocess_s", pa.float64()),
        pa.field("qx_delivery_s", pa.float64()),

        # ── Section 4: Local Timing ──
        pa.field("local_submit_s", pa.float64()),
        pa.field("local_wait_s", pa.float64()),
        pa.field("wall_total_s", pa.float64()),

        # ── Section 5: Derived (QPU only, nullable) ──
        pa.field("per_circuit_qpu_s", pa.float64()),
        pa.field("per_shot_us", pa.float64()),

        # ── Section 6: Context (QPU only, nullable) ──
        pa.field("queue_length_before", pa.int32()),
        pa.field("calibration_set_id", pa.string()),
        pa.field("active_reset_cycles", pa.string()),
        pa.field("dd_mode", pa.string()),
        pa.field("heralding_mode", pa.string()),
        pa.field("max_circuit_duration_s", pa.float64()),  # provisional

        # ── Section 7: Retry (RED-DIRECTIVE-QPU-CONFIG §5) ──
        pa.field("retry_attempts", pa.int32()),            # 1=first try, 2+=retried, null=sim

        # ── Section 8: Infrastructure ──
        pa.field("device", pa.string()),
        pa.field("partition", pa.string()),
        pa.field("slurm_job_id", pa.string()),
        pa.field("node", pa.string()),
    ])


def export_benchmark_to_parquet(
    timing_records: list[Any],
    sweep_metadata: dict[str, Any],
    output_path: str | Path,
    *,
    packing_metadata: list[dict[str, Any]] | None = None,
) -> int:
    """Export per-batch benchmark data to a 36-column Parquet file.

    Args:
        timing_records: List of QPUJobTiming dataclasses (from
            IqmQpuBackend.get_batch_timings()) or dicts with equivalent
            keys for simulator sweeps.
        sweep_metadata: Dict with keys: sweep_id, framework_version,
            mode ("qpu" or "simulator"), device, partition, shots,
            circuit_width. Used to populate identity and infrastructure
            columns.
        output_path: Path for the output .parquet file.
        packing_metadata: Optional per-batch packing info. List of dicts
            with keys: n_placements, packing_rounds, packing_algorithm,
            qubit_utilization_mean. If None or shorter than timing_records,
            defaults are used.

    Returns:
        Number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = _benchmark_schema()
    output_path = Path(output_path)

    # Collect infrastructure context once
    sweep_id = sweep_metadata.get("sweep_id", "unknown")
    fw_version = sweep_metadata.get("framework_version", "")
    mode = sweep_metadata.get("mode", "simulator")
    device = sweep_metadata.get("device", "unknown")
    partition = sweep_metadata.get("partition", os.getenv("SLURM_JOB_PARTITION", "unknown"))
    slurm_job_id = os.getenv("SLURM_JOB_ID")
    node = os.getenv("SLURMD_NODENAME") or os.getenv("HOSTNAME")
    default_shots = sweep_metadata.get("shots", 4096)
    default_width = sweep_metadata.get("circuit_width", 0)

    if not timing_records:
        # Write an empty table with the correct schema
        table = pa.table({f.name: pa.array([], type=f.type) for f in schema}, schema=schema)
        pq.write_table(table, str(output_path), compression="snappy")
        return 0

    # Build column arrays
    rows: dict[str, list] = {f.name: [] for f in schema}

    for batch_idx, rec in enumerate(timing_records):
        # Support both dataclass (QPUJobTiming) and dict
        def _get(key, default=None):
            if isinstance(rec, dict):
                return rec.get(key, default)
            return getattr(rec, key, default)

        now_utc = datetime.now(timezone.utc).isoformat()

        # Packing metadata for this batch
        pm = {}
        if packing_metadata and batch_idx < len(packing_metadata):
            pm = packing_metadata[batch_idx]

        n_circuits = _get("n_circuits", 0)
        shots = _get("shots", default_shots)
        qpu_execute = _get("qpu_execute_s")

        # ── Identity ──
        rows["sweep_id"].append(sweep_id)
        rows["job_id"].append(_get("job_id"))
        rows["batch_index"].append(batch_idx)
        rows["framework_version"].append(fw_version)
        rows["timestamp_utc"].append(now_utc)
        rows["mode"].append(mode)

        # ── Batch ──
        rows["n_circuits"].append(n_circuits)
        rows["shots"].append(shots)
        rows["total_shots"].append(n_circuits * shots if n_circuits and shots else 0)
        rows["circuit_width"].append(pm.get("circuit_width", default_width))
        rows["n_placements"].append(pm.get("n_placements", n_circuits))
        rows["packing_rounds"].append(pm.get("packing_rounds", 1))
        rows["packing_algorithm"].append(pm.get("packing_algorithm", "none"))
        rows["qubit_utilization_mean"].append(
            pm.get("qubit_utilization_mean", 0.0)
        )

        # ── Server Timing (nullable — None for simulator) ──
        rows["qx_overhead_s"].append(_get("qx_overhead_s"))
        rows["qx_compile_s"].append(_get("qx_compile_s"))
        rows["qpu_queue_s"].append(_get("qpu_queue_s"))
        rows["qpu_execute_s"].append(qpu_execute)
        rows["qx_postprocess_s"].append(_get("qx_postprocess_s"))
        rows["qx_delivery_s"].append(_get("qx_delivery_s"))

        # ── Local Timing ──
        rows["local_submit_s"].append(_get("local_submit_s", 0.0))
        rows["local_wait_s"].append(_get("local_wait_s", 0.0))
        rows["wall_total_s"].append(_get("wall_total_s", 0.0))

        # ── Derived ──
        if qpu_execute is not None and n_circuits and n_circuits > 0:
            per_circuit = qpu_execute / n_circuits
            per_shot = (per_circuit / shots * 1e6) if shots > 0 else None
        else:
            per_circuit = None
            per_shot = None
        rows["per_circuit_qpu_s"].append(per_circuit)
        rows["per_shot_us"].append(per_shot)

        # ── Context (from job payload — QPU only) ──
        rows["queue_length_before"].append(_get("queue_length_before"))
        rows["calibration_set_id"].append(_get("calibration_set_id"))
        rows["active_reset_cycles"].append(_get("active_reset_cycles"))
        rows["dd_mode"].append(_get("dd_mode"))
        rows["heralding_mode"].append(_get("heralding_mode"))
        rows["max_circuit_duration_s"].append(_get("max_circuit_duration_s"))

        # ── Retry (RED-DIRECTIVE-QPU-CONFIG §5) ──
        rows["retry_attempts"].append(_get("retry_attempts"))

        # ── Infrastructure ──
        rows["device"].append(device)
        rows["partition"].append(partition)
        rows["slurm_job_id"].append(slurm_job_id)
        rows["node"].append(node)

    # Build PyArrow arrays with correct types
    arrays = []
    for f in schema:
        col = rows[f.name]
        arrays.append(pa.array(col, type=f.type))

    table = pa.table(
        {f.name: arrays[i] for i, f in enumerate(schema)},
        schema=schema,
    )
    pq.write_table(table, str(output_path), compression="snappy")

    return len(timing_records)


def make_simulator_timing_records(
    timing_json: dict[str, Any],
) -> list[dict[str, Any]]:
    """Convert sweep_timing.json data into benchmark-compatible records.

    For simulator sweeps, there's one "batch" representing the entire
    sweep. QPU-specific fields are None.

    Args:
        timing_json: Parsed contents of sweep_timing.json.

    Returns:
        List of dicts compatible with export_benchmark_to_parquet().
    """
    phases = timing_json.get("phases", {})
    total_s = timing_json.get("total_elapsed_seconds", 0.0)

    return [{
        "job_id": None,
        "n_circuits": timing_json.get("total_circuits", 0),
        "shots": timing_json.get("shots", 0),
        "wall_total_s": total_s,
        "local_submit_s": 0.0,
        "local_wait_s": total_s,
        # Server-side fields: None for simulator
        "qx_overhead_s": None,
        "qx_compile_s": None,
        "qpu_queue_s": None,
        "qpu_execute_s": None,
        "qx_postprocess_s": None,
        "qx_delivery_s": None,
        "queue_length_before": None,
        "calibration_set_id": None,
        "active_reset_cycles": None,
        "dd_mode": None,
        "heralding_mode": None,
        "max_circuit_duration_s": None,
        # Retry — null for simulator
        "retry_attempts": None,
    }]
