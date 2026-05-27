# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""CLI wrapper for running a BYO sweep YAML through sweep_engine.run_sweep_from_dict.

Usage:
    python3 -m lumi_hpc_qc.sweep.run_sweep <sweep.yaml> [--device CPU|GPU]

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

    result = run_sweep_from_dict(config_dict, device=args.device)
    print(f"Sweep complete: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
