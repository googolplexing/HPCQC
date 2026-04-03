# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Synthetic calibration generator — CLI tool for creating perturbed calibrations.

Reads a real calibration JSON, applies perturbations (T1/T2 scaling, readout
degradation, qubit poisoning, uniform noise, future device projection), and
writes a synthetic calibration JSON with full provenance metadata.

The output JSON is in IQM v2 format — directly consumable by the existing
noise model builder, twin simulator, and sweep engine. The `_synthetic_metadata`
block records exactly how the calibration was generated.

RED-SPEC-002 §10 — Synthetic Calibration Tools

Usage:
    # Scale T1 by 0.7 (30% degradation)
    python -m lumi_hpc_qc.data.tools.perturb_calibration \\
        examples/q50_calibration_20260330.json \\
        --perturb t1 --factor 0.7 \\
        --output calibrations/synthetic_t1_degraded_30pct.json

    # Degrade readout fidelity by 10%
    python -m lumi_hpc_qc.data.tools.perturb_calibration \\
        examples/q50_calibration_20260330.json \\
        --perturb readout --factor 0.9 \\
        --output calibrations/synthetic_readout_degraded.json

    # Poison a specific qubit
    python -m lumi_hpc_qc.data.tools.perturb_calibration \\
        examples/q50_calibration_20260330.json \\
        --degrade-qubit QB23 \\
        --output calibrations/synthetic_qb23_poisoned.json

    # Set all qubits to identical T1/T2 (remove noise diversity)
    python -m lumi_hpc_qc.data.tools.perturb_calibration \\
        examples/q50_calibration_20260330.json \\
        --uniform-noise --t1 30.0 --t2 15.0 \\
        --output calibrations/synthetic_uniform.json

    # Project to a future better device (all params improved 2×)
    python -m lumi_hpc_qc.data.tools.perturb_calibration \\
        examples/q50_calibration_20260330.json \\
        --improve-all --factor 2.0 \\
        --output calibrations/synthetic_future_2x.json

    # Programmatic usage:
    from lumi_hpc_qc.data.tools.perturb_calibration import generate_synthetic
    result = generate_synthetic(
        "examples/q50_calibration_20260330.json",
        {"scale_t1": 0.7},
        "output/synthetic.json",
    )
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def generate_synthetic(
    input_path: str,
    perturbation: dict[str, Any],
    output_path: str,
) -> dict[str, Any]:
    """Generate a synthetic calibration from a real one.

    Thin wrapper around SyntheticAdapter.perturb() + save().

    Args:
        input_path: Path to real calibration JSON (IQM v2 format).
        perturbation: Perturbation spec dict (see SyntheticAdapter.perturb).
        output_path: Path to write synthetic calibration JSON.

    Returns:
        Dict with generation summary:
          - input: str
          - output: str
          - perturbation: dict
          - num_qubits: int
          - num_gates: int
          - warnings: list[str]
    """
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    from lumi_hpc_qc.plugins.calibration_adapters.synthetic import SyntheticAdapter

    adapter = IQMv2Adapter()
    synth = SyntheticAdapter(adapter)

    # Load base calibration
    base_cal = adapter.load(input_path)

    # Apply perturbation
    perturbed = synth.perturb(base_cal, perturbation)

    # Validate
    warnings = synth.validate(perturbed)

    # Save
    synth.save(perturbed, output_path)

    return {
        "input": input_path,
        "output": output_path,
        "perturbation": perturbation,
        "num_qubits": perturbed.num_qubits,
        "num_gates": len(perturbed.gates),
        "is_synthetic": perturbed.is_synthetic,
        "description": perturbed.synthetic_perturbation,
        "warnings": warnings,
    }


def generate_batch(
    input_path: str,
    perturbations: list[dict[str, Any]],
    output_dir: str,
    *,
    name_template: str = "synthetic_{index:03d}.json",
) -> list[dict[str, Any]]:
    """Generate multiple synthetic calibrations from one base.

    Useful for creating a sweep across noise regimes:
        perturbations = [
            {"scale_t1": f} for f in [0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]
        ]

    Args:
        input_path: Path to real calibration JSON.
        perturbations: List of perturbation spec dicts.
        output_dir: Directory for output files.
        name_template: Filename template with {index} placeholder.

    Returns:
        List of generation summaries.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = []

    for i, perturbation in enumerate(perturbations):
        filename = name_template.format(index=i)
        output_path = str(Path(output_dir) / filename)
        result = generate_synthetic(input_path, perturbation, output_path)
        results.append(result)

    return results


def _build_perturbation_from_args(args: argparse.Namespace) -> dict[str, Any]:
    """Convert CLI arguments into a perturbation spec dict."""
    perturbation: dict[str, Any] = {}

    if args.perturb and args.factor is not None:
        key_map = {
            "t1": "scale_t1",
            "t2": "scale_t2",
            "readout": "scale_readout",
            "readout_fidelity": "scale_readout",
            "gate_error": "scale_gate_error",
            "gate_fidelity": "scale_gate_fidelity",
        }
        perturb_key = key_map.get(args.perturb)
        if perturb_key is None:
            print(f"Unknown perturbation target: {args.perturb}")
            print(f"Available: {', '.join(sorted(key_map.keys()))}")
            sys.exit(1)
        perturbation[perturb_key] = args.factor

    if args.degrade_qubit:
        perturbation["poison_qubit"] = args.degrade_qubit

    if args.uniform_noise:
        if args.t1 is not None:
            perturbation["uniform_t1"] = args.t1
        if args.t2 is not None:
            perturbation["uniform_t2"] = args.t2
        if args.readout_value is not None:
            perturbation["uniform_readout"] = args.readout_value

    if args.improve_all:
        if args.factor is None:
            print("--improve-all requires --factor F")
            sys.exit(1)
        perturbation["improve_all"] = args.factor

    if not perturbation:
        print("No perturbation specified. Use --perturb, --degrade-qubit, "
              "--uniform-noise, or --improve-all.")
        sys.exit(1)

    return perturbation


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="perturb_calibration",
        description="Generate synthetic QPU calibration files with controlled perturbations.",
    )

    parser.add_argument(
        "input",
        help="Path to the real calibration JSON file (IQM v2 format)",
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path for the synthetic calibration output JSON",
    )

    # Scaling perturbations
    parser.add_argument(
        "--perturb",
        choices=["t1", "t2", "readout", "readout_fidelity",
                 "gate_error", "gate_fidelity"],
        help="Parameter to perturb (used with --factor)",
    )
    parser.add_argument(
        "--factor", "-f",
        type=float,
        help="Scaling factor (e.g., 0.7 = 30%% degradation, 1.5 = 50%% improvement)",
    )

    # Qubit poisoning
    parser.add_argument(
        "--degrade-qubit",
        metavar="QB_NAME",
        help="Set a specific qubit to severely degraded values (e.g., QB23)",
    )

    # Uniform noise
    parser.add_argument(
        "--uniform-noise",
        action="store_true",
        help="Set all qubits to identical noise values (use with --t1, --t2)",
    )
    parser.add_argument(
        "--t1", type=float, help="Uniform T1 value in microseconds",
    )
    parser.add_argument(
        "--t2", type=float, help="Uniform T2 value in microseconds",
    )
    parser.add_argument(
        "--readout-value", type=float, help="Uniform readout fidelity (0–1)",
    )

    # Future device projection
    parser.add_argument(
        "--improve-all",
        action="store_true",
        help="Scale all parameters toward better values (use with --factor > 1)",
    )

    # Verbosity
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output except errors",
    )

    args = parser.parse_args()

    # Build perturbation spec
    perturbation = _build_perturbation_from_args(args)

    # Generate
    try:
        result = generate_synthetic(args.input, perturbation, args.output)
    except FileNotFoundError:
        print(f"Error: calibration file not found: {args.input}")
        sys.exit(1)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if not args.quiet:
        print(f"Synthetic calibration generated:")
        print(f"  Input:        {result['input']}")
        print(f"  Output:       {result['output']}")
        print(f"  Perturbation: {result['description']}")
        print(f"  Qubits:       {result['num_qubits']}")
        print(f"  Gates:        {result['num_gates']}")
        if result["warnings"]:
            print(f"  Warnings:")
            for w in result["warnings"]:
                print(f"    - {w}")


if __name__ == "__main__":
    main()
