# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""AI/ML training data export — flatten experiment JSON into tabular formats.

Converts the hierarchical ExperimentRecord JSON into flat CSV suitable
for ML pipelines (energy prediction, optimizer hyperparameter tuning,
circuit architecture search).

Usage:
    from lumi_hpc_qc.data.export import export_training_data
    export_training_data("results/byo/exp_12345.json", "training_data.csv")

Or from CLI:
    python -m lumi_hpc_qc.data.export results/*.json --output training_data.csv

Output columns:
    experiment_id, model, ansatz, optimizer, gradient, backend, precision,
    num_qubits, num_params, iteration, energy, best_energy_so_far,
    elapsed_s, gradient_norm, is_best, exact_energy, relative_error,
    param_0, param_1, ..., param_N
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


def _load_json(path: str) -> dict:
    """Load JSON with numpy array reconstruction."""
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


def export_training_data(
    input_paths: list[str] | str,
    output_path: str,
    include_params: bool = True,
    max_params: int = 128,
) -> int:
    """Export experiment JSON files to flat CSV.

    Args:
        input_paths: Path(s) to experiment result JSON files.
        output_path: Output CSV file path.
        include_params: Whether to include individual parameter columns.
        max_params: Maximum number of parameter columns to include.

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

        # Extract experiment-level fields
        config = data.get("config", {})
        exp_id = data.get("experiment_id", config.get("experiment_id", ""))
        model = config.get("model", "")
        ansatz = config.get("ansatz", "")
        optimizer = config.get("optimizer", "")
        gradient = config.get("gradient", "")
        backend = config.get("backend", "")
        precision = config.get("precision", "")
        num_qubits = config.get("num_qubits", 0)

        exact_energy = data.get("exact_energy")

        # Extract iterations
        iterations = data.get("iterations", [])
        if not iterations:
            # Single-result format (CircuitSubmissionWorkflow)
            result = data.get("result", {})
            if result:
                row = {
                    "experiment_id": exp_id,
                    "model": model,
                    "ansatz": ansatz,
                    "optimizer": optimizer,
                    "gradient": gradient,
                    "backend": backend,
                    "precision": precision,
                    "num_qubits": num_qubits,
                    "num_params": 0,
                    "iteration": 0,
                    "energy": result.get("fun", 0.0),
                    "best_energy_so_far": result.get("fun", 0.0),
                    "elapsed_s": 0.0,
                    "gradient_norm": None,
                    "is_best": True,
                    "exact_energy": exact_energy,
                    "relative_error": None,
                }
                if exact_energy is not None and abs(exact_energy) > 1e-10:
                    row["relative_error"] = abs(row["energy"] - exact_energy) / abs(exact_energy) * 100
                rows.append(row)
            continue

        best_so_far = float('inf')
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

            row = {
                "experiment_id": exp_id,
                "model": model,
                "ansatz": ansatz,
                "optimizer": optimizer,
                "gradient": gradient,
                "backend": backend,
                "precision": precision,
                "num_qubits": num_qubits,
                "num_params": num_params,
                "iteration": it.get("iteration", 0),
                "energy": energy,
                "best_energy_so_far": best_so_far,
                "elapsed_s": it.get("elapsed_s", 0.0),
                "gradient_norm": it.get("gradient_norm"),
                "is_best": it.get("is_best", False),
                "exact_energy": exact_energy,
                "relative_error": None,
            }

            if exact_energy is not None and abs(exact_energy) > 1e-10:
                row["relative_error"] = abs(energy - exact_energy) / abs(exact_energy) * 100

            # Add individual parameters as columns
            if include_params and params is not None:
                for j in range(min(len(params), max_params)):
                    row[f"param_{j}"] = float(params[j])

            rows.append(row)

    if not rows:
        print("  No data to export.")
        return 0

    # Write CSV
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(rows[0].keys())

    # Ensure all param columns are in fieldnames (union across all rows)
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    param_keys = sorted([k for k in all_keys if k.startswith("param_")])
    for k in param_keys:
        if k not in fieldnames:
            fieldnames.append(k)

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Exported {len(rows)} rows to {output_path}")
    print(f"  Columns: {len(fieldnames)} ({len(param_keys)} parameter columns)")
    return len(rows)


def export_summary(input_paths: list[str] | str, output_path: str) -> int:
    """Export one row per experiment — summary statistics only.

    Useful for hyperparameter analysis and experiment comparison.
    """
    if isinstance(input_paths, str):
        input_paths = [input_paths]

    rows = []
    for path in input_paths:
        try:
            data = _load_json(path)
        except (json.JSONDecodeError, FileNotFoundError):
            continue

        config = data.get("config", {})
        result = data.get("result", {})
        timing = data.get("timing", {})

        exact = data.get("exact_energy")
        best_e = result.get("fun", 0.0)

        row = {
            "experiment_id": data.get("experiment_id", ""),
            "model": config.get("model", ""),
            "ansatz": config.get("ansatz", ""),
            "optimizer": config.get("optimizer", ""),
            "gradient": config.get("gradient", ""),
            "backend": config.get("backend", ""),
            "precision": config.get("precision", ""),
            "num_qubits": config.get("num_qubits", 0),
            "num_params": len(result.get("x", [])) if result.get("x") else 0,
            "total_iterations": result.get("nit", 0),
            "total_function_evals": result.get("nfev", 0),
            "best_energy": best_e,
            "exact_energy": exact,
            "absolute_error": abs(best_e - exact) if exact else None,
            "relative_error_pct": abs(best_e - exact) / abs(exact) * 100 if exact and abs(exact) > 1e-10 else None,
            "total_wall_time_s": timing.get("total_s", 0.0),
            "optimizer_maxiter": config.get("optimizer_params", {}).get("maxiter", 0),
            "ansatz_reps": config.get("ansatz_params", {}).get("reps", 0),
            "initializer": config.get("initializer", ""),
        }
        rows.append(row)

    if not rows:
        return 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  Exported {len(rows)} experiment summaries to {output_path}")
    return len(rows)


# CLI entry point
if __name__ == "__main__":
    import sys
    import glob

    if len(sys.argv) < 2:
        print("Usage: python -m lumi_hpc_qc.data.export <result_json_files...> [--output FILE] [--summary]")
        print("  --output FILE    Output CSV path (default: training_data.csv)")
        print("  --summary        Export one row per experiment instead of per-iteration")
        print("  --no-params      Exclude individual parameter columns")
        sys.exit(1)

    args = sys.argv[1:]
    output = "training_data.csv"
    summary_mode = False
    include_params = True
    paths = []

    i = 0
    while i < len(args):
        if args[i] == "--output" and i + 1 < len(args):
            output = args[i + 1]
            i += 2
        elif args[i] == "--summary":
            summary_mode = True
            i += 1
        elif args[i] == "--no-params":
            include_params = False
            i += 1
        else:
            # Expand globs
            expanded = glob.glob(args[i])
            paths.extend(expanded if expanded else [args[i]])
            i += 1

    if not paths:
        print("No input files found.")
        sys.exit(1)

    print(f"Processing {len(paths)} result file(s)...")
    if summary_mode:
        export_summary(paths, output)
    else:
        export_training_data(paths, output, include_params=include_params)
