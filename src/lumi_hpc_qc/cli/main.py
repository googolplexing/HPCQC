# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""CLI entry point for lumi-hpc-qc.

Commands:
  lumi-vqa run --config experiment.yaml       Run experiment (Mode A)
  lumi-vqa submit --config experiment.yaml    Generate + submit SLURM job
  lumi-vqa status --job-id 12345              Check job status
  lumi-vqa resume --experiment-id abc123      Resume from checkpoint
  lumi-vqa results --experiment-id abc123     Print convergence summary
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        prog="lumi-vqa",
        description="LUMI HPC + Quantum Computing experiment framework",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- run ---
    p_run = subparsers.add_parser("run", help="Run experiment interactively (Mode A)")
    p_run.add_argument("--config", required=True, help="Path to experiment YAML config")
    p_run.add_argument("--override", nargs="*", help="Override config: key=value")

    # --- submit ---
    p_submit = subparsers.add_parser("submit", help="Generate SLURM script and submit")
    p_submit.add_argument("--config", required=True, help="Path to experiment YAML config")
    p_submit.add_argument("--mode", default="interactive",
                          choices=["interactive", "automated"],
                          help="Execution mode (default: interactive)")
    p_submit.add_argument("--dry-run", action="store_true",
                          help="Generate script but don't submit")

    # --- status ---
    p_status = subparsers.add_parser("status", help="Check SLURM job status")
    p_status.add_argument("--job-id", required=True, help="SLURM job ID")

    # --- resume ---
    p_resume = subparsers.add_parser("resume", help="Resume from checkpoint")
    p_resume.add_argument("--config", required=True, help="Path to experiment YAML config")
    p_resume.add_argument("--experiment-id", help="Experiment ID to resume")
    p_resume.add_argument("--checkpoint", help="Explicit checkpoint file path")

    # --- results ---
    p_results = subparsers.add_parser("results", help="Print experiment results summary")
    p_results.add_argument("--experiment-id", help="Experiment ID")
    p_results.add_argument("--output-dir", default="results", help="Results directory")
    p_results.add_argument("--format", default="human",
                           choices=["human", "json", "benchmark", "training"],
                           help="Output format")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "run":
        _cmd_run(args)
    elif args.command == "submit":
        _cmd_submit(args)
    elif args.command == "status":
        _cmd_status(args)
    elif args.command == "resume":
        _cmd_resume(args)
    elif args.command == "results":
        _cmd_results(args)


def _cmd_run(args: argparse.Namespace) -> None:
    """Run experiment interactively."""
    from lumi_hpc_qc.cli.config_loader import load_config
    from lumi_hpc_qc.orchestration.controller import Controller

    overrides = _parse_overrides(args.override) if args.override else None
    config = load_config(args.config, overrides)

    controller = Controller()
    record = controller.run_interactive(config)

    print(f"\nExperiment {record.experiment_id} complete.")
    if record.convergence:
        print(f"  Best energy: {record.convergence.best_energy:.8f}")
        if record.convergence.relative_error_pct is not None:
            print(f"  Relative error: {record.convergence.relative_error_pct:.4f}%")
    if record.timing:
        print(record.timing.to_human_readable())


def _cmd_submit(args: argparse.Namespace) -> None:
    """Generate SLURM script and optionally submit."""
    from lumi_hpc_qc.cli.config_loader import load_config
    from lumi_hpc_qc.cli.slurm_templates import write_slurm_script

    config = load_config(args.config)
    config.mode = args.mode

    script_path = write_slurm_script(config)
    print(f"SLURM script written: {script_path}")

    if not args.dry_run:
        from lumi_hpc_qc.orchestration.scheduler import SlurmScheduler
        scheduler = SlurmScheduler()
        job_id = scheduler.submit(script_path)
        print(f"Submitted: job {job_id}")
    else:
        print("(dry run — not submitted)")


def _cmd_status(args: argparse.Namespace) -> None:
    """Check job status."""
    from lumi_hpc_qc.orchestration.scheduler import SlurmScheduler
    scheduler = SlurmScheduler()
    status = scheduler.status(args.job_id)
    print(f"Job {args.job_id}: {status}")


def _cmd_resume(args: argparse.Namespace) -> None:
    """Resume from checkpoint."""
    from lumi_hpc_qc.cli.config_loader import load_config
    from lumi_hpc_qc.orchestration.checkpoint import CheckpointManager
    from lumi_hpc_qc.orchestration.controller import Controller

    config = load_config(args.config)

    # Find checkpoint
    if args.checkpoint:
        checkpoint_path = args.checkpoint
    else:
        mgr = CheckpointManager(config.checkpoint.directory)
        exp_id = args.experiment_id or config.experiment_id
        checkpoint_path = mgr.exists(exp_id)
        if not checkpoint_path:
            print(f"No checkpoint found for experiment {exp_id}")
            sys.exit(1)

    print(f"Resuming from: {checkpoint_path}")
    controller = Controller()
    record = controller.resume(checkpoint_path, config)
    print(f"Experiment {record.experiment_id} resumed and complete.")


def _cmd_results(args: argparse.Namespace) -> None:
    """Print experiment results."""
    from lumi_hpc_qc.data.result_store import load_json
    from pathlib import Path
    import json

    output_dir = Path(args.output_dir)
    if args.experiment_id:
        candidates = list(output_dir.rglob(f"*{args.experiment_id}*_result.json"))
    else:
        candidates = list(output_dir.rglob("*_result.json"))

    if not candidates:
        print("No results found.")
        sys.exit(1)

    for path in sorted(candidates):
        data = load_json(path)
        if args.format == "json":
            print(json.dumps(data, indent=2))
        elif args.format == "human":
            _print_human(data)
        elif args.format == "benchmark":
            if "timing" in data and data["timing"]:
                print(json.dumps(data["timing"], indent=2))
        elif args.format == "training":
            _print_training(data)


def _print_human(data: dict) -> None:
    """Print human-readable experiment summary."""
    print(f"  Experiment: {data.get('experiment_id', 'N/A')}")
    conv = data.get("convergence", {})
    if conv:
        print(f"  Best energy:    {conv.get('best_energy', 'N/A')}")
        print(f"  Exact energy:   {conv.get('exact_ground_energy', 'N/A')}")
        print(f"  Relative error: {conv.get('relative_error_pct', 'N/A')}%")
        print(f"  Iterations:     {conv.get('total_iterations', 'N/A')}")


def _print_training(data: dict) -> None:
    """Print flat training-format record."""
    import json
    record = {}
    config = data.get("config", {})
    record["model"] = config.get("model", "")
    record["ansatz"] = config.get("ansatz", "")
    record["num_qubits"] = config.get("num_qubits", 0)
    record["backend"] = config.get("backend", "")
    record["precision"] = config.get("precision", "")
    conv = data.get("convergence", {})
    record.update({f"conv_{k}": v for k, v in conv.items()})
    timing = data.get("timing", {})
    if timing:
        for phase, dur in timing.get("phases", {}).items():
            record[f"timing_{phase}_s"] = dur
    print(json.dumps(record, indent=2))


def _parse_overrides(override_list: list[str]) -> dict:
    """Parse CLI overrides like 'precision=single model=heisenberg'."""
    overrides = {}
    for item in override_list:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        # Try numeric conversion
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                pass
        overrides[key] = value
    return overrides


if __name__ == "__main__":
    main()
