# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""CLI wrapper for running a BYO sweep YAML through sweep_engine.run_sweep_from_dict.

Usage:
    python3 -m lumi_hpc_qc.sweep.run_sweep <sweep.yaml> [--device CPU|GPU]
                                            [--output-dir DIR]

--output-dir overrides the YAML's `output_dir` for this invocation only (the
file on disk is untouched). It lets a caller point an unmodified config at a
clean, run-specific workdir — e.g. the W1.6 gate runs the canonical
floquet_dtc_q10_sweep.yaml into sweep_output/w1_gate without editing it.

The YAML schema is the one parse_sweep_config / run_sweep_from_dict accept:
top-level `sweep:` key with `experiments`, `calibrations`, `execution`, etc.
See examples/byo/floquet_byo_sweep.yaml and floquet_dtc_q10_sweep.yaml for
worked examples, and docs/MIGRATION_FloquetDTC.md for the distillation pattern.

Exit code:
    0  sweep completed successfully
    1  argv / config / runtime error (message printed to stderr)
"""
from __future__ import annotations

import argparse
import sys

import yaml

from lumi_hpc_qc.sweep.sweep_engine import run_sweep_from_dict


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python3 -m lumi_hpc_qc.sweep.run_sweep",
        description="Run a BYO sweep YAML through the sweep engine.",
    )
    parser.add_argument(
        "config",
        help="Path to a sweep YAML (top-level 'sweep:' key).",
    )
    parser.add_argument(
        "--device",
        default="CPU",
        choices=["CPU", "GPU"],
        help="Execution device (default: CPU).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Override the YAML's output_dir for this run only (the file is not "
            "modified). Useful for directing an unmodified config at a clean, "
            "run-specific workdir."
        ),
    )
    parser.add_argument(
        "--physical-qubits",
        default=None,
        help=(
            "Override the BYO experiment(s)' physical_qubits for this run only "
            "(in-memory; the YAML is not modified). Comma-separated qubit names "
            "supplying ONE explicit placement, e.g. 'QB1,QB2,QB5'; bypasses the "
            "placement solver (PLACEMENT-1). Logical qubit i maps to the i-th "
            "name. Does not affect grid expansion / task_ids / the campaign "
            "manifest (placements multiply execution units, not grid tasks)."
        ),
    )
    args = parser.parse_args(argv)

    try:
        with open(args.config, encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except OSError as exc:
        print(f"error: could not read config file {args.config!r}: {exc}", file=sys.stderr)
        return 1
    except yaml.YAMLError as exc:
        print(f"error: invalid YAML in {args.config!r}: {exc}", file=sys.stderr)
        return 1

    if not isinstance(config_dict, dict) or "sweep" not in config_dict:
        print(
            f"error: {args.config!r} must be a YAML mapping with a top-level 'sweep:' key",
            file=sys.stderr,
        )
        return 1

    # --output-dir override: mutate the same sub-dict parse_sweep_config reads
    # from (it does `yaml_dict.get("sweep", yaml_dict)`), so the override takes
    # effect whether or not the config is wrapped in a top-level `sweep:` key.
    # In-memory only; the YAML on disk is never written back.
    if args.output_dir is not None:
        sweep_section = config_dict.get("sweep", config_dict)
        sweep_section["output_dir"] = args.output_dir

    # --physical-qubits override (PLACEMENT-1): pin the BYO experiment(s) to ONE
    # explicit placement, bypassing the solver, for this run only. In-memory;
    # the YAML on disk is never written back. Comma-separated names -> one
    # placement (a list of names), which parse_sweep_config normalizes via
    # _parse_physical_qubits. This does NOT touch grid expansion / task_ids /
    # the campaign manifest: placements multiply execution UNITS inside
    # _execute_byo_group, not the grid TASKS that expand_grid enumerates, so the
    # task set (and thus the manifest key) is identical with or without it.
    if args.physical_qubits is not None:
        names = [q.strip() for q in args.physical_qubits.split(",") if q.strip()]
        if not names:
            print("error: --physical-qubits given but no qubit names parsed",
                  file=sys.stderr)
            return 1
        sweep_section = config_dict.get("sweep", config_dict)
        experiments = sweep_section.get("experiments", []) or []
        applied = 0
        for exp in experiments:
            if isinstance(exp, dict) and exp.get("type") == "byo_circuit":
                exp["physical_qubits"] = names      # one placement; parser wraps it
                applied += 1
        if applied == 0:
            print("error: --physical-qubits given but no byo_circuit experiment "
                  "found to apply it to", file=sys.stderr)
            return 1
        print(f"[run_sweep] --physical-qubits: pinned {applied} byo experiment(s) "
              f"to placement {names}")

    result = run_sweep_from_dict(config_dict, device=args.device)
    print(f"Sweep complete: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
