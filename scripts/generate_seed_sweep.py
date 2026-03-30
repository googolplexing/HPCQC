#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Generate multi-seed sweep configs for statistical characterization.

RED-SPEC-001-v1.1 §7.3: Run the same experiment with different random
seeds to characterize optimizer convergence variance.

Usage:
    # Generate 20-seed sweep from a base config:
    python scripts/generate_seed_sweep.py configs/q50bench_tfim_2q_noiseless.yaml -n 20

    # Generate and show output directory:
    python scripts/generate_seed_sweep.py configs/q50bench_tfim_4q_noiseless.yaml -n 20 --output-dir configs/seed_sweep

Output: One YAML config per seed, identical to base except initializer_params.seed.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


def generate_seed_sweep(base_config_path: str, num_seeds: int, output_dir: str) -> list[str]:
    """Generate N configs varying only in initializer seed.

    Args:
        base_config_path: Path to the base YAML config.
        num_seeds: Number of seed variants to generate.
        output_dir: Directory for output configs.

    Returns:
        List of generated file paths.
    """
    with open(base_config_path) as f:
        base = yaml.safe_load(f)

    model_name = Path(base_config_path).stem
    os.makedirs(output_dir, exist_ok=True)
    generated = []

    for i in range(num_seeds):
        seed = 42 + i  # deterministic, reproducible seed sequence
        config = dict(base)
        config["initializer_params"] = dict(base.get("initializer_params", {}))
        config["initializer_params"]["seed"] = seed

        # Unique output directory per seed to avoid result collisions
        config["output_dir"] = f"results/seed_sweep/{model_name}/seed_{seed:04d}"

        filename = f"{model_name}_seed{seed:04d}.yaml"
        filepath = os.path.join(output_dir, filename)

        header = (
            f"# Auto-generated seed sweep: seed={seed} ({i+1}/{num_seeds})\n"
            f"# Base config: {base_config_path}\n"
            f"#\n"
        )
        with open(filepath, "w") as f:
            f.write(header)
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        generated.append(filepath)

    return generated


def generate_sweep_launcher(configs: list[str], output_path: str, base_name: str) -> None:
    """Generate a SLURM launcher script that submits all seed configs.

    Args:
        configs: List of config file paths.
        output_path: Path for the launcher script.
        base_name: Base model name for job naming.
    """
    lines = [
        "#!/bin/bash",
        "# Auto-generated seed sweep launcher",
        "# Submit all seed configs as separate SLURM jobs",
        "",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "${SCRIPT_DIR}/../env.sh"',
        "",
        f'echo "=== Seed Sweep: {base_name} ({len(configs)} seeds) ==="',
        'echo "Date: $(date)"',
        'echo ""',
        "",
        "JOB_IDS=()",
        "",
    ]

    for i, cfg in enumerate(configs):
        seed = 42 + i
        lines.append(f'# Seed {seed}')
        lines.append(
            f'JOB_ID=$(sbatch --job-name=seed_{seed:04d} '
            f'--partition=${{HPCQC_GPU_PARTITION}} --time=00:30:00 '
            f'--nodes=1 --gpus-per-node=8 --ntasks-per-node=1 --cpus-per-task=56 '
            f'--output=slurm_logs/seed_sweep_{base_name}_{seed:04d}.o%j '
            f'--error=slurm_logs/seed_sweep_{base_name}_{seed:04d}.e%j '
            f'--wrap="export SINGULARITYENV_PROJECT_DIR=${{HPCQC_ROOT}} && '
            f'export SINGULARITYENV_PYTHONPATH=${{HPCQC_ROOT}}/src && '
            f'srun --cpu-bind=${{HPCQC_GPU_MASK}} ${{HPCQC_GPU_WRAPPER}} ${{HPCQC_GPU_CONTAINER}} '
            f'python -c \\"import sys; sys.path.insert(0,\'${{HPCQC_ROOT}}/src\'); '
            f'from lumi_hpc_qc.orchestration.workflow import VQEWorkflow; '
            f'from lumi_hpc_qc.cli.config_loader import load_config; '
            f'config = load_config(\'${{HPCQC_ROOT}}/{cfg}\'); '
            f'VQEWorkflow().run(config)\\"" '
            f'2>&1 | grep -oP \'\\d+$\')'
        )
        lines.append(f'echo "  seed {seed}: job $JOB_ID"')
        lines.append(f'JOB_IDS+=("$JOB_ID")')
        lines.append("")

    lines.extend([
        f'echo ""',
        f'echo "Submitted ${{#JOB_IDS[@]}} jobs"',
        f'echo "Monitor: squeue --me"',
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    os.chmod(output_path, 0o755)


def main():
    parser = argparse.ArgumentParser(description="Generate multi-seed sweep configs")
    parser.add_argument("base_config", help="Path to base YAML config")
    parser.add_argument("-n", "--num-seeds", type=int, default=20,
                        help="Number of seeds (default: 20)")
    parser.add_argument("--output-dir", default="configs/seed_sweep",
                        help="Output directory for configs")
    parser.add_argument("--launcher", action="store_true",
                        help="Also generate SLURM launcher script")
    args = parser.parse_args()

    configs = generate_seed_sweep(args.base_config, args.num_seeds, args.output_dir)
    base_name = Path(args.base_config).stem.replace("q50bench_", "").replace("_noiseless", "")

    print(f"Generated {len(configs)} seed configs in {args.output_dir}/")
    for c in configs[:3]:
        print(f"  {c}")
    if len(configs) > 3:
        print(f"  ... ({len(configs) - 3} more)")

    if args.launcher:
        launcher_path = os.path.join(args.output_dir, f"launch_sweep_{base_name}.sh")
        generate_sweep_launcher(configs, launcher_path, base_name)
        print(f"Launcher: {launcher_path}")


if __name__ == "__main__":
    main()
