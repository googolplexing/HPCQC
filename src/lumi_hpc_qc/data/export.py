# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Multi-format export — convert experiment JSON to ML-consumable datasets.

Phase D (RED-SPEC-001) adds five new export formats alongside the existing
CSV exporter. All formats read the same JSON produced by ExperimentTracker
and write to an output directory alongside the source JSON.

Formats
-------
CSV    existing — flat per-iteration table (param_0 … param_N columns)
Parquet  new   — columnar ML training data (pyarrow, no param_N explosion)
HDF5     new   — hierarchical numerical data (h5py)
JSONL    new   — one JSON object per iteration, append-friendly
NPZ      new   — numpy arrays: energy_trajectory, param_trajectory, gradient_norms
QPY      new   — Qiskit circuit serialisation (ansatz only, written at setup time)

Usage
-----
Single experiment, all formats:

    from lumi_hpc_qc.data.export import export_all
    export_all("results/tfim_4q/abc123_result.json", "dataset/")

Specific formats:

    from lumi_hpc_qc.data.export import export_parquet, export_hdf5
    export_parquet(["results/exp1.json", "results/exp2.json"], "dataset/training.parquet")
    export_hdf5(["results/exp1.json"], "dataset/experiments.h5")

Existing CSV exporter (unchanged):

    from lumi_hpc_qc.data.export import export_training_data
    export_training_data("results/exp.json", "dataset/training.csv")

CLI:

    python -m lumi_hpc_qc.data.export results/*.json --output dataset/ --format all
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_json(path: str) -> dict:
    """Load a result JSON with numpy array reconstruction."""
    with open(path) as f:
        data = json.load(f)

    def _reconstruct(obj):
        if isinstance(obj, dict) and obj.get("__ndarray__"):
            import numpy as np
            return np.array(obj["data"], dtype=obj.get("dtype", "float64"))
        if isinstance(obj, dict):
            return {k: _reconstruct(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_reconstruct(v) for v in obj]
        return obj

    return _reconstruct(data)


def _experiment_fields(data: dict) -> dict:
    """Extract the experiment-level fields shared across all formats.

    Returns a flat dict of scalar fields that describe the experiment
    as a whole — one value per experiment, not one per iteration.
    """
    config  = data.get("config") or {}
    conv    = data.get("convergence") or {}
    cm      = data.get("circuit_metrics") or {}
    noise   = data.get("noise_config") or {}
    em      = data.get("error_mitigation_applied") or {}
    timing  = data.get("timing") or {}

    # Phase D: noiseless_tier — None for non-sweep runs
    tier = data.get("noiseless_tier")

    return {
        # Identity
        "experiment_id":            data.get("experiment_id", ""),
        "schema_version":           data.get("schema_version", "1.0.0"),
        "created_at":               data.get("created_at", ""),
        # Config
        "model":                    config.get("model", ""),
        "ansatz":                   config.get("ansatz", ""),
        "optimizer":                config.get("optimizer", ""),
        "gradient":                 config.get("gradient", ""),
        "initializer":              config.get("initializer", ""),
        "backend":                  config.get("backend", ""),
        "precision":                config.get("precision", ""),
        "num_qubits":               config.get("num_qubits", 0),
        "ansatz_reps":              (config.get("ansatz_params") or {}).get("reps", 0),
        "optimizer_maxiter":        (config.get("optimizer_params") or {}).get("maxiter", 0),
        # Convergence
        "best_energy":              conv.get("best_energy"),
        "exact_energy":             conv.get("exact_ground_energy"),
        "absolute_error":           conv.get("absolute_error"),
        "relative_error_pct":       conv.get("relative_error_pct"),
        "total_iterations":         conv.get("total_iterations", 0),
        "optimizer_converged":      conv.get("optimizer_converged", False),
        # Circuit metrics (Phase B — may be None for noiseless runs)
        "circuit_depth_pre":        cm.get("pre_transpilation_depth"),
        "circuit_depth_post":       cm.get("post_transpilation_depth"),
        "cx_count_pre":             cm.get("pre_transpilation_cx_count"),
        "cx_count_post":            cm.get("post_transpilation_cx_count"),
        "swap_count":               cm.get("swap_count"),
        "coupling_map_source":      cm.get("coupling_map_source"),
        # Noise config (Phase B — None for noiseless)
        "noise_channels_active":    noise.get("channels_active", "none")
                                    if isinstance(noise.get("channels_active"), str)
                                    else _channels_str(noise.get("channels_active")),
        # Error mitigation (Phase D)
        "mitigation_readout":       em.get("readout", False),
        "mitigation_zne":           em.get("zne", False),
        # Sweep classification (Phase D — None for non-sweep runs)
        "noiseless_tier":           tier,
        # Timing
        "wall_time_s":              (timing.get("total_s") or
                                     (timing.get("phases") or {}).get("total", None)),
    }


def _channels_str(channels: dict | None) -> str:
    """Convert noise channels dict to a compact comma-separated string."""
    if not channels:
        return "none"
    active = [k for k, v in channels.items() if v]
    return ",".join(active) if active else "none"


# ---------------------------------------------------------------------------
# CSV export — existing functions, enriched with Phase D columns
# ---------------------------------------------------------------------------

def export_training_data(
    input_paths: list[str] | str,
    output_path: str,
    include_params: bool = True,
    max_params: int = 128,
) -> int:
    """Export experiment JSON files to flat per-iteration CSV.

    Each row is one optimizer iteration. Experiment-level fields
    (model, ansatz, circuit metrics, noise config, noiseless_tier)
    are repeated on every row so the CSV is self-contained for ML
    training without requiring joins.

    Args:
        input_paths:    Path(s) to experiment result JSON files.
        output_path:    Output CSV file path.
        include_params: Whether to include individual parameter columns.
        max_params:     Maximum number of parameter columns (default 128).

    Returns:
        Number of rows written.
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    rows = []

    for path in input_paths:
        try:
            data = _load_json(path)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  WARNING: skipping {path}: {e}")
            continue

        exp = _experiment_fields(data)
        iterations = data.get("iterations", [])

        if not iterations:
            # CircuitSubmissionWorkflow — single result, no iteration loop
            result = data.get("result") or {}
            if result:
                row = {**exp,
                       "num_params": 0, "iteration": 0,
                       "energy": result.get("fun", 0.0),
                       "best_energy_so_far": result.get("fun", 0.0),
                       "elapsed_s": 0.0, "gradient_norm": None, "is_best": True,
                       "relative_error": None}
                if exp["exact_energy"] and abs(exp["exact_energy"]) > 1e-10:
                    row["relative_error"] = (
                        abs(row["energy"] - exp["exact_energy"])
                        / abs(exp["exact_energy"]) * 100
                    )
                rows.append(row)
            continue

        best_so_far = float("inf")
        num_params = 0

        for it in iterations:
            energy = it.get("energy", 0.0)
            if energy < best_so_far:
                best_so_far = energy

            params = it.get("parameters")
            if params is not None:
                import numpy as np
                if isinstance(params, list):
                    params = np.array(params)
                num_params = len(params)

            row = {**exp,
                   "num_params":        num_params,
                   "iteration":         it.get("iteration", 0),
                   "energy":            energy,
                   "best_energy_so_far": best_so_far,
                   "elapsed_s":         it.get("elapsed_s", 0.0),
                   "gradient_norm":     it.get("gradient_norm"),
                   "is_best":           it.get("is_best", False),
                   "relative_error":    None}

            if exp["exact_energy"] and abs(exp["exact_energy"]) > 1e-10:
                row["relative_error"] = (
                    abs(energy - exp["exact_energy"])
                    / abs(exp["exact_energy"]) * 100
                )

            if include_params and params is not None:
                for j in range(min(len(params), max_params)):
                    row[f"param_{j}"] = float(params[j])

            rows.append(row)

    if not rows:
        print("  No data to export.")
        return 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())
    all_keys: set = set()
    for r in rows:
        all_keys.update(r.keys())
    for k in sorted(k for k in all_keys if k.startswith("param_")):
        if k not in fieldnames:
            fieldnames.append(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV: {len(rows)} rows → {output_path}")
    return len(rows)


def export_summary(input_paths: list[str] | str, output_path: str) -> int:
    """Export one row per experiment — summary statistics only."""
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    rows = []
    for path in input_paths:
        try:
            data = _load_json(path)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        rows.append(_experiment_fields(data))

    if not rows:
        return 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  CSV summary: {len(rows)} experiments → {output_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# JSONL export — one JSON object per iteration, append-friendly
# ---------------------------------------------------------------------------

def export_jsonl(
    input_paths: list[str] | str,
    output_path: str,
) -> int:
    """Export per-iteration data as JSON Lines (one object per line).

    Each line is a complete JSON object combining experiment-level fields
    with the iteration fields. Compatible with Hugging Face datasets,
    streaming pipelines, and log aggregators.

    Args:
        input_paths: Path(s) to experiment result JSON files.
        output_path: Output .jsonl file path.

    Returns:
        Number of lines written.
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    total = 0

    with open(output_path, "w") as f:
        for path in input_paths:
            try:
                data = _load_json(path)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"  WARNING: skipping {path}: {e}")
                continue

            exp = _experiment_fields(data)

            for it in data.get("iterations", []):
                params = it.get("parameters")
                if params is not None and hasattr(params, "tolist"):
                    params = params.tolist()

                record = {
                    **exp,
                    "iteration":     it.get("iteration", 0),
                    "energy":        it.get("energy", 0.0),
                    "gradient_norm": it.get("gradient_norm"),
                    "elapsed_s":     it.get("elapsed_s", 0.0),
                    "is_best":       it.get("is_best", False),
                    "parameters":    params,
                    "timestamp":     it.get("timestamp", ""),
                }
                f.write(json.dumps(record, default=str) + "\n")
                total += 1

    print(f"  JSONL: {total} lines → {output_path}")
    return total


# ---------------------------------------------------------------------------
# NPZ export — numpy arrays for energy/parameter trajectories
# ---------------------------------------------------------------------------

def export_npz(
    input_paths: list[str] | str,
    output_dir: str,
) -> int:
    """Export per-experiment numpy arrays as compressed .npz files.

    Each experiment gets its own .npz file containing:
      energy_trajectory  shape (N,)       — energy at each iteration
      param_trajectory   shape (N, P)     — parameters at each iteration
      gradient_norms     shape (N,)       — gradient norm (NaN where unavailable)
      metadata           dict as JSON str — experiment-level fields

    Args:
        input_paths: Path(s) to experiment result JSON files.
        output_dir:  Directory to write .npz files into.

    Returns:
        Number of .npz files written.
    """
    import numpy as np

    if isinstance(input_paths, str):
        input_paths = [input_paths]

    os.makedirs(output_dir, exist_ok=True)
    written = 0

    for path in input_paths:
        try:
            data = _load_json(path)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  WARNING: skipping {path}: {e}")
            continue

        exp_id = data.get("experiment_id", "unknown")
        iterations = data.get("iterations", [])
        if not iterations:
            continue

        energies = np.array([it.get("energy", float("nan")) for it in iterations])

        grad_norms = np.array([
            it.get("gradient_norm") if it.get("gradient_norm") is not None
            else float("nan")
            for it in iterations
        ])

        # Parameter trajectory — shape (N, P)
        params_list = []
        for it in iterations:
            p = it.get("parameters")
            if p is not None:
                if hasattr(p, "tolist"):
                    p = p.tolist()
                params_list.append(p)
            else:
                params_list.append([])

        # Pad to uniform shape if param count varies (shouldn't happen, but safe)
        max_p = max((len(p) for p in params_list), default=0)
        if max_p > 0:
            param_arr = np.full((len(params_list), max_p), float("nan"))
            for i, p in enumerate(params_list):
                param_arr[i, :len(p)] = p
        else:
            param_arr = np.empty((len(iterations), 0))

        exp_fields = _experiment_fields(data)
        metadata_json = json.dumps(exp_fields, default=str)

        out_path = os.path.join(output_dir, f"{exp_id}.npz")
        np.savez_compressed(
            out_path,
            energy_trajectory=energies,
            param_trajectory=param_arr,
            gradient_norms=grad_norms,
            metadata=np.array(metadata_json),
        )
        written += 1

    print(f"  NPZ: {written} files → {output_dir}/")
    return written


# ---------------------------------------------------------------------------
# HDF5 export — hierarchical numerical data
# ---------------------------------------------------------------------------

def _derive_hdf5_group_name(data: dict, json_path: str) -> str:
    """Build human-readable HDF5 group name per RED-RESP-V19 §1 Mod 2.

    Format: {model}-{qubits:02d}q-{mode}-{id_short}-seed{seed:04d}
    Dashes between fields, underscores within fields.
    """
    config = data.get("config") or {}
    model = config.get("model", "unknown")
    num_qubits = config.get("num_qubits", 0)
    exp_id = data.get("experiment_id", "unknown")

    # Extract first 8 hex chars from experiment_id (format: {hex12}_{slurm_or_interactive})
    id_short = exp_id.split("_")[0][:8] if "_" in exp_id else exp_id[:8]

    # Derive seed from initializer_params
    init_params = config.get("initializer_params") or {}
    seed = init_params.get("seed", 0)

    # Derive mode from backend_params
    bp = config.get("backend_params") or {}
    noise_file = bp.get("noise_model_file")
    noise_channels = bp.get("noise_channels")
    cm_source = bp.get("coupling_map_source", "full")
    shots = bp.get("shots", 0)

    if not noise_file and shots == 0:
        mode = "noiseless"
    elif not noise_file and cm_source == "calibration":
        mode = "topology_noiseless"
    elif noise_file and noise_channels:
        # Check if all channels are active
        active = [k for k, v in noise_channels.items() if v]
        if len(active) == 0:
            mode = "controlled_noiseless"
        elif len(active) == 1:
            mode = f"noise_{active[0]}"
        else:
            mode = "noise_full"
    elif noise_file:
        mode = "noise_full"
    else:
        mode = "noiseless"

    return f"{model}-{num_qubits:02d}q-{mode}-{id_short}-seed{seed:04d}"


def _find_measurement_stats_sidecar(json_path: str, data: dict) -> str | None:
    """Locate the measurement stats JSONL sidecar for an experiment."""
    # Check if the result JSON references a sidecar
    sidecar_name = data.get("measurement_stats_sidecar")
    if sidecar_name:
        sidecar_path = os.path.join(os.path.dirname(json_path), sidecar_name)
        if os.path.exists(sidecar_path):
            return sidecar_path

    # Fallback: look for {experiment_id}_measurement_stats.jsonl
    exp_id = data.get("experiment_id", "")
    if exp_id:
        sidecar_path = os.path.join(
            os.path.dirname(json_path),
            f"{exp_id}_measurement_stats.jsonl"
        )
        if os.path.exists(sidecar_path):
            return sidecar_path

    return None


def export_hdf5(
    input_paths: list[str] | str,
    output_path: str,
) -> int:
    """Export experiments to a single HDF5 file.

    Hierarchy (RED-RESP-V19 Modifications 2+3):
      /experiments/{model}-{qubits:02d}q-{mode}-{id_short}-seed{seed:04d}/
        energy_trajectory     dataset float64 (N,)
        param_trajectory      dataset float64 (N, P)
        gradient_norms        dataset float64 (N,)
        metadata/             group of scalar attributes
          model, ansatz, best_energy, noiseless_tier, ...
          experiment_id = full UUID (for programmatic lookup)
        measurement_stats     dataset string (M,)   [if sidecar exists]
          attrs: grouping_algorithm = "qwc"
          attrs: interval = 10
          attrs: num_entries = M

    Multiple experiments from a sweep are all stored in one file,
    making cross-experiment analysis straightforward:

        import h5py, json
        with h5py.File("experiments.h5", "r") as f:
            for name in f["experiments"]:
                energy = f[f"experiments/{name}/energy_trajectory"][:]
                stats = [json.loads(s) for s in f[f"experiments/{name}/measurement_stats"][:]]

    Args:
        input_paths: Path(s) to experiment result JSON files.
        output_path: Output .h5 file path.

    Returns:
        Number of experiments written.
    """
    import h5py
    import numpy as np

    if isinstance(input_paths, str):
        input_paths = [input_paths]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    written = 0

    with h5py.File(output_path, "w") as hf:
        exps_group = hf.create_group("experiments")

        for path in input_paths:
            try:
                data = _load_json(path)
            except (json.JSONDecodeError, FileNotFoundError) as e:
                print(f"  WARNING: skipping {path}: {e}")
                continue

            iterations = data.get("iterations", [])
            if not iterations:
                continue

            # Human-readable group name (RED-RESP-V19 Mod 2+3)
            group_name = _derive_hdf5_group_name(data, path)

            # Handle collision (append suffix if name exists)
            final_name = group_name
            suffix = 2
            while final_name in exps_group:
                final_name = f"{group_name}-{suffix}"
                suffix += 1

            grp = exps_group.create_group(final_name)

            # Trajectory datasets
            energies = np.array([it.get("energy", float("nan"))
                                  for it in iterations], dtype=np.float64)
            grp.create_dataset("energy_trajectory", data=energies,
                               compression="gzip", compression_opts=4)

            grad_norms = np.array([
                it.get("gradient_norm") if it.get("gradient_norm") is not None
                else float("nan")
                for it in iterations
            ], dtype=np.float64)
            grp.create_dataset("gradient_norms", data=grad_norms,
                               compression="gzip", compression_opts=4)

            # Parameter trajectory
            params_list = []
            for it in iterations:
                p = it.get("parameters")
                if p is not None:
                    params_list.append(p if isinstance(p, list) else p.tolist())
                else:
                    params_list.append([])

            max_p = max((len(p) for p in params_list), default=0)
            if max_p > 0:
                param_arr = np.full((len(params_list), max_p),
                                    float("nan"), dtype=np.float64)
                for i, p in enumerate(params_list):
                    param_arr[i, :len(p)] = p
                grp.create_dataset("param_trajectory", data=param_arr,
                                   compression="gzip", compression_opts=4)

            # Metadata as HDF5 attributes — scalar fields + identity attrs
            exp_fields = _experiment_fields(data)
            meta_grp = grp.create_group("metadata")
            for k, v in exp_fields.items():
                if v is None:
                    meta_grp.attrs[k] = "null"
                elif isinstance(v, (int, float, bool, str)):
                    meta_grp.attrs[k] = v
                else:
                    meta_grp.attrs[k] = str(v)

            # Top-level group attributes for quick programmatic access
            config = data.get("config") or {}
            init_params = config.get("initializer_params") or {}
            grp.attrs["experiment_id"] = data.get("experiment_id", "")
            grp.attrs["model"] = config.get("model", "")
            grp.attrs["num_qubits"] = config.get("num_qubits", 0)
            grp.attrs["seed"] = init_params.get("seed", 0)

            # V19: embed measurement stats sidecar as flat string array
            sidecar_path = _find_measurement_stats_sidecar(path, data)
            if sidecar_path:
                with open(sidecar_path) as sf:
                    lines = [line.strip() for line in sf if line.strip()]
                if lines:
                    # h5py special string type for variable-length strings
                    dt = h5py.string_dtype(encoding="utf-8")
                    ds = grp.create_dataset(
                        "measurement_stats", data=lines, dtype=dt,
                        compression="gzip", compression_opts=4,
                    )
                    ds.attrs["grouping_algorithm"] = "qwc"
                    ds.attrs["interval"] = (
                        config.get("measurement_stats_interval", 10)
                    )
                    ds.attrs["num_entries"] = len(lines)

            written += 1

    print(f"  HDF5: {written} experiments → {output_path}")
    return written


# ---------------------------------------------------------------------------
# Parquet export — columnar ML training data
# ---------------------------------------------------------------------------

def export_parquet(
    input_paths: list[str] | str,
    output_path: str,
) -> int:
    """Export per-iteration data to Parquet (primary ML format).

    Uses pyarrow (available in container after GREEN-REQ-001 rebuild).
    Parameters are stored as a single list column rather than 128 separate
    param_N columns — this is the key advantage over the CSV format for
    ML pipelines that don't need individual parameter columns.

    Schema:
      experiment_id      string
      model              string
      ansatz             string
      ...all _experiment_fields scalar columns...
      noiseless_tier     int32 (nullable)
      iteration          int32
      energy             double
      gradient_norm      double (nullable)
      elapsed_s          double
      is_best            bool
      parameters         list<double>   ← replaces param_0..param_N

    Args:
        input_paths: Path(s) to experiment result JSON files.
        output_path: Output .parquet file path.

    Returns:
        Number of rows written.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import numpy as np

    if isinstance(input_paths, str):
        input_paths = [input_paths]

    # Collect all rows as plain Python dicts first, then build Arrow table
    rows = []

    for path in input_paths:
        try:
            data = _load_json(path)
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"  WARNING: skipping {path}: {e}")
            continue

        exp = _experiment_fields(data)

        for it in data.get("iterations", []):
            params = it.get("parameters")
            if params is not None:
                if hasattr(params, "tolist"):
                    params = params.tolist()
                elif not isinstance(params, list):
                    params = list(params)
            else:
                params = []

            row = {
                **exp,
                "iteration":     it.get("iteration", 0),
                "energy":        it.get("energy", 0.0),
                "gradient_norm": it.get("gradient_norm"),
                "elapsed_s":     it.get("elapsed_s", 0.0),
                "is_best":       it.get("is_best", False),
                "parameters":    params,
            }
            rows.append(row)

    if not rows:
        print("  No data to export.")
        return 0

    # Build Arrow schema
    # Scalar fields from _experiment_fields
    scalar_fields = [
        pa.field("experiment_id",       pa.string()),
        pa.field("schema_version",      pa.string()),
        pa.field("created_at",          pa.string()),
        pa.field("model",               pa.string()),
        pa.field("ansatz",              pa.string()),
        pa.field("optimizer",           pa.string()),
        pa.field("gradient",            pa.string()),
        pa.field("initializer",         pa.string()),
        pa.field("backend",             pa.string()),
        pa.field("precision",           pa.string()),
        pa.field("num_qubits",          pa.int32()),
        pa.field("ansatz_reps",         pa.int32()),
        pa.field("optimizer_maxiter",   pa.int32()),
        pa.field("best_energy",         pa.float64()),
        pa.field("exact_energy",        pa.float64()),
        pa.field("absolute_error",      pa.float64()),
        pa.field("relative_error_pct",  pa.float64()),
        pa.field("total_iterations",    pa.int32()),
        pa.field("optimizer_converged", pa.bool_()),
        pa.field("circuit_depth_pre",   pa.int32()),
        pa.field("circuit_depth_post",  pa.int32()),
        pa.field("cx_count_pre",        pa.int32()),
        pa.field("cx_count_post",       pa.int32()),
        pa.field("swap_count",          pa.int32()),
        pa.field("coupling_map_source", pa.string()),
        pa.field("noise_channels_active", pa.string()),
        pa.field("mitigation_readout",  pa.bool_()),
        pa.field("mitigation_zne",      pa.bool_()),
        pa.field("noiseless_tier",      pa.int32()),   # nullable
        pa.field("wall_time_s",         pa.float64()),
        # Per-iteration fields
        pa.field("iteration",           pa.int32()),
        pa.field("energy",              pa.float64()),
        pa.field("gradient_norm",       pa.float64()),  # nullable
        pa.field("elapsed_s",           pa.float64()),
        pa.field("is_best",             pa.bool_()),
        pa.field("parameters",          pa.list_(pa.float64())),
    ]
    schema = pa.schema(scalar_fields)

    # Convert rows to columnar arrays, handling None → null safely
    def _col(key, dtype):
        vals = [r.get(key) for r in rows]
        if pa.types.is_list(dtype):
            return pa.array(vals, type=dtype)
        if pa.types.is_boolean(dtype):
            return pa.array([bool(v) if v is not None else False for v in vals],
                            type=dtype)
        if pa.types.is_integer(dtype):
            return pa.array([int(v) if v is not None else None for v in vals],
                            type=dtype)
        if pa.types.is_floating(dtype):
            return pa.array([float(v) if v is not None else None for v in vals],
                            type=dtype)
        return pa.array([str(v) if v is not None else "" for v in vals],
                        type=dtype)

    arrays = [_col(f.name, f.type) for f in schema]
    table = pa.table(dict(zip([f.name for f in schema], arrays)),
                     schema=schema)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")

    print(f"  Parquet: {len(rows)} rows, {len(schema)} columns → {output_path}")
    return len(rows)


# ---------------------------------------------------------------------------
# export_all — convenience wrapper: writes all formats at once
# ---------------------------------------------------------------------------

def export_all(
    input_paths: list[str] | str,
    output_dir: str,
    base_name: str = "experiments",
) -> dict[str, int]:
    """Export to all formats at once.

    Writes into output_dir:
      {base_name}.parquet
      {base_name}.h5
      {base_name}.jsonl
      {base_name}_summary.csv
      {base_name}_iterations.csv
      npz/  (one .npz per experiment)

    Args:
        input_paths: Path(s) to experiment result JSON files.
        output_dir:  Output directory (created if needed).
        base_name:   Base filename for outputs (default: "experiments").

    Returns:
        Dict mapping format name to number of rows/files written.
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    os.makedirs(output_dir, exist_ok=True)
    p = Path(output_dir)
    results = {}

    print(f"Exporting {len(input_paths)} experiment(s) to {output_dir}/")

    try:
        results["parquet"] = export_parquet(
            input_paths, str(p / f"{base_name}.parquet"))
    except ImportError:
        print("  Parquet: skipped (pyarrow not available)")
        results["parquet"] = 0

    try:
        results["hdf5"] = export_hdf5(
            input_paths, str(p / f"{base_name}.h5"))
    except ImportError:
        print("  HDF5: skipped (h5py not available)")
        results["hdf5"] = 0

    results["jsonl"] = export_jsonl(
        input_paths, str(p / f"{base_name}.jsonl"))

    results["npz"] = export_npz(
        input_paths, str(p / "npz"))

    results["csv_iterations"] = export_training_data(
        input_paths, str(p / f"{base_name}_iterations.csv"))

    results["csv_summary"] = export_summary(
        input_paths, str(p / f"{base_name}_summary.csv"))

    total = sum(results.values())
    print(f"Export complete. {total} total records written.")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) < 2:
        print("Usage: python -m lumi_hpc_qc.data.export <result_json_files...> "
              "[--output PATH] [--format FORMAT]")
        print()
        print("  --output PATH    Output file or directory (default: training_data.csv)")
        print("  --format FORMAT  One of: csv, summary, parquet, hdf5, jsonl, npz, all")
        print("                   Default: csv")
        print("  --no-params      Exclude parameter columns from CSV output")
        sys.exit(1)

    args = sys.argv[1:]
    output = "training_data.csv"
    fmt = "csv"
    include_params = True
    paths = []

    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]; i += 2
        elif args[i] == "--format" and i + 1 < len(args):
            fmt = args[i + 1]; i += 2
        elif args[i] == "--no-params":
            include_params = False; i += 1
        else:
            expanded = glob.glob(args[i])
            paths.extend(expanded if expanded else [args[i]]); i += 1

    if not paths:
        print("No input files found.")
        sys.exit(1)

    print(f"Processing {len(paths)} result file(s)...")

    dispatch = {
        "csv":     lambda: export_training_data(paths, output, include_params),
        "summary": lambda: export_summary(paths, output),
        "parquet": lambda: export_parquet(paths, output),
        "hdf5":    lambda: export_hdf5(paths, output),
        "jsonl":   lambda: export_jsonl(paths, output),
        "npz":     lambda: export_npz(paths, output),
        "all":     lambda: export_all(paths, output),
    }

    if fmt not in dispatch:
        print(f"Unknown format '{fmt}'. Choose: {', '.join(dispatch)}")
        sys.exit(1)

    dispatch[fmt]()
