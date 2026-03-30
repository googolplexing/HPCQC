#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Reproducibility demonstration — RED-SPEC-001-v1.1 §7.4.

Re-runs a completed experiment from its provenance metadata on a fresh
LUMI allocation. Compares energy trajectories using statistical
equivalence (max |ΔE| < 0.01% per iteration), NOT bitwise identity.

The numpy downgrade (2.4.3 → 2.2.6) means bitwise reproduction against
the original v1.0.0b3 run is impossible. This is expected and documented.

Usage:
    # Compare two experiment result JSONs:
    python scripts/reproducibility_check.py results/original.json results/reproduced.json

    # Generate a reproduction config from an existing result:
    python scripts/reproducibility_check.py results/original.json --generate-config
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def load_experiment(path: str) -> dict:
    """Load an experiment result JSON."""
    with open(path) as f:
        return json.load(f)


def extract_energy_trajectory(experiment: dict) -> list[float]:
    """Extract the per-iteration energy trajectory."""
    iterations = experiment.get("iterations", [])
    return [it.get("energy", float("nan")) for it in iterations]


def compare_trajectories(
    original: list[float],
    reproduced: list[float],
    threshold_pct: float = 0.01,
) -> dict:
    """Compare two energy trajectories.

    Args:
        original: Energy values from original run.
        reproduced: Energy values from reproduction run.
        threshold_pct: Maximum allowed |ΔE|/|E| per iteration (%).

    Returns:
        Dict with comparison results.
    """
    n_orig = len(original)
    n_repr = len(reproduced)
    n_compare = min(n_orig, n_repr)

    if n_compare == 0:
        return {"pass": False, "reason": "No iterations to compare"}

    deltas = []
    relative_deltas = []
    max_delta = 0.0
    max_delta_iter = 0
    failures = []

    for i in range(n_compare):
        e_orig = original[i]
        e_repr = reproduced[i]

        if np.isnan(e_orig) or np.isnan(e_repr):
            continue

        delta = abs(e_repr - e_orig)
        deltas.append(delta)

        if abs(e_orig) > 1e-12:
            rel_delta_pct = delta / abs(e_orig) * 100
        else:
            rel_delta_pct = 0.0

        relative_deltas.append(rel_delta_pct)

        if delta > max_delta:
            max_delta = delta
            max_delta_iter = i

        if rel_delta_pct > threshold_pct:
            failures.append({
                "iteration": i,
                "original": e_orig,
                "reproduced": e_repr,
                "delta": delta,
                "relative_pct": rel_delta_pct,
            })

    passed = len(failures) == 0

    return {
        "pass": passed,
        "iterations_compared": n_compare,
        "iterations_original": n_orig,
        "iterations_reproduced": n_repr,
        "max_absolute_delta": max_delta,
        "max_delta_iteration": max_delta_iter,
        "mean_absolute_delta": np.mean(deltas) if deltas else 0,
        "max_relative_delta_pct": max(relative_deltas) if relative_deltas else 0,
        "mean_relative_delta_pct": np.mean(relative_deltas) if relative_deltas else 0,
        "threshold_pct": threshold_pct,
        "failures": failures,
        "num_failures": len(failures),
    }


def generate_reproduction_config(experiment: dict, output_path: str) -> None:
    """Generate a YAML config to reproduce an experiment.

    Extracts the configuration from the experiment record and writes
    a YAML file that can be used to re-run the same experiment.
    """
    config = experiment.get("config", {})
    if not config:
        print("ERROR: No config found in experiment record")
        return

    # Extract provenance info
    provenance = experiment.get("provenance", {})
    original_numpy = "unknown"
    if provenance:
        modules = provenance.get("imported_modules", {})
        original_numpy = modules.get("numpy", "unknown")

    header = (
        f"# Reproduction config — generated from experiment {experiment.get('experiment_id', 'unknown')}\n"
        f"# Original numpy version: {original_numpy}\n"
        f"# Current numpy version will differ — use statistical equivalence for comparison\n"
        f"# Threshold: max |ΔE| < 0.01% per iteration\n"
        f"#\n"
    )

    # Write as YAML
    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Reproduction config written to: {output_path}")
    print(f"Original numpy: {original_numpy}")
    print(f"Run with: python -c 'from lumi_hpc_qc.orchestration.workflow import VQEWorkflow; "
          f"from lumi_hpc_qc.cli.config_loader import load_config; "
          f"VQEWorkflow().run(load_config(\"{output_path}\"))'")


def print_report(comparison: dict, original_path: str, reproduced_path: str) -> None:
    """Print a formatted reproducibility report."""
    print(f"\n{'='*60}")
    print(f"REPRODUCIBILITY REPORT")
    print(f"{'='*60}")
    print(f"Original:   {original_path}")
    print(f"Reproduced: {reproduced_path}")
    print(f"Threshold:  max |ΔE| < {comparison['threshold_pct']}% per iteration")
    print()
    print(f"Iterations compared: {comparison['iterations_compared']}")
    print(f"  Original:   {comparison['iterations_original']}")
    print(f"  Reproduced: {comparison['iterations_reproduced']}")
    print()
    print(f"Max absolute ΔE:  {comparison['max_absolute_delta']:.2e} "
          f"(iteration {comparison['max_delta_iteration']})")
    print(f"Mean absolute ΔE: {comparison['mean_absolute_delta']:.2e}")
    print(f"Max relative ΔE:  {comparison['max_relative_delta_pct']:.6f}%")
    print(f"Mean relative ΔE: {comparison['mean_relative_delta_pct']:.6f}%")
    print()

    if comparison["pass"]:
        print(f"RESULT: PASS — all iterations within {comparison['threshold_pct']}% threshold")
    else:
        print(f"RESULT: FAIL — {comparison['num_failures']} iterations exceed threshold")
        for f in comparison["failures"][:5]:
            print(f"  Iter {f['iteration']}: "
                  f"orig={f['original']:.8f}, repr={f['reproduced']:.8f}, "
                  f"Δ={f['relative_pct']:.4f}%")

    print()
    print("NOTE: The numpy version changed from 2.4.3 to 2.2.6 during the")
    print("container rebuild (GREEN-REQ-001). Floating-point differences in")
    print("reduction order and eigenvalue computation are expected. The")
    print("comparison uses statistical equivalence, not bitwise identity.")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Reproducibility check")
    parser.add_argument("original", help="Path to original experiment JSON")
    parser.add_argument("reproduced", nargs="?", help="Path to reproduced experiment JSON")
    parser.add_argument("--generate-config", action="store_true",
                        help="Generate reproduction config instead of comparing")
    parser.add_argument("--threshold", type=float, default=0.01,
                        help="Max |ΔE|/|E| threshold in %% (default: 0.01)")
    parser.add_argument("--output", default="configs/reproduction.yaml",
                        help="Output path for generated config")
    args = parser.parse_args()

    original = load_experiment(args.original)

    if args.generate_config:
        generate_reproduction_config(original, args.output)
        return

    if not args.reproduced:
        parser.error("Provide reproduced experiment JSON, or use --generate-config")

    reproduced = load_experiment(args.reproduced)

    traj_orig = extract_energy_trajectory(original)
    traj_repr = extract_energy_trajectory(reproduced)

    comparison = compare_trajectories(traj_orig, traj_repr, args.threshold)
    print_report(comparison, args.original, args.reproduced)

    sys.exit(0 if comparison["pass"] else 1)


if __name__ == "__main__":
    main()
