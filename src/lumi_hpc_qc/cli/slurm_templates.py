# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""SLURM batch script generator.

All generated scripts source env.sh for container/wrapper/account paths.
No hardcoded absolute paths — edit env.sh once to change everything.
"""

from __future__ import annotations

from pathlib import Path

from lumi_hpc_qc.types import ExperimentConfig


def generate_child_vqe_script(
    config: ExperimentConfig,
    config_yaml_path: str,
    runner_script: str = "tests/vqe_runner.py",
    resume_checkpoint: str | None = None,
) -> str:
    """Generate child VQE job script that sources env.sh for all paths."""
    slurm = config.slurm
    job_name = f"vqe_{config.model}_{config.num_qubits or 'N'}q"

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --account={slurm.account}",
        f"#SBATCH --partition={slurm.partition}",
        f"#SBATCH --time={slurm.walltime}",
        f"#SBATCH --nodes={slurm.nodes}",
    ]
    if slurm.gpus_per_node > 0:
        lines.append(f"#SBATCH --gpus-per-node={slurm.gpus_per_node}")
    lines += [
        "#SBATCH --ntasks-per-node=1",
        "#SBATCH --cpus-per-task=56",
        '#SBATCH --output=slurm_logs/%x.o%j',
        '#SBATCH --error=slurm_logs/%x.e%j',
        "",
        "# Source central environment config",
        'SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"',
        'source "${SCRIPT_DIR}/../env.sh"',
        "",
        'export SINGULARITYENV_PROJECT_DIR="${HPCQC_ROOT}"',
        f'export SINGULARITYENV_CONFIG_PATH="${{HPCQC_ROOT}}/{config_yaml_path}"',
    ]

    if resume_checkpoint:
        lines.append(f'export SINGULARITYENV_RESUME_CHECKPOINT="{resume_checkpoint}"')

    lines += [
        "",
        'mkdir -p "${HPCQC_ROOT}/slurm_logs"',
        "",
        "SLURM_START_EPOCH=$(date +%s)",
        f'echo "=== {job_name} ==="',
        'echo "Job ${SLURM_JOB_ID} on ${SLURM_NODELIST}"',
        "export SINGULARITYENV_SLURM_START_EPOCH=$SLURM_START_EPOCH",
        "",
        'srun --cpu-bind=${HPCQC_GPU_MASK} \\',
        '  ${HPCQC_GPU_WRAPPER} ${HPCQC_GPU_CONTAINER} \\',
        f'  python ${{HPCQC_ROOT}}/{runner_script}',
        "",
        'echo "Completed: $(date), wall: $(( $(date +%s) - SLURM_START_EPOCH ))s"',
    ]
    return "\n".join(lines)


def write_slurm_script(config: ExperimentConfig, output_dir: str = "slurm") -> str:
    """Generate and write a SLURM script to file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    filename = f"{config.model}_{config.num_qubits or 'N'}q.sh"
    filepath = output_path / filename
    script = generate_child_vqe_script(config, f"configs/{config.model}.yaml")
    with open(filepath, "w") as f:
        f.write(script)
    filepath.chmod(0o755)
    return str(filepath)


def estimate_walltime(config: ExperimentConfig) -> str:
    """Estimate walltime from experiment parameters."""
    nq = config.num_qubits or 12
    maxiter = config.optimizer_params.get("maxiter", 200)
    if config.backend == "aer_gpu":
        sec_per_eval = {12: 0.007, 18: 0.05, 24: 0.5, 30: 5, 36: 30}.get(nq, 0.007 * 2**(nq-12))
    elif config.backend == "aer_cpu":
        sec_per_eval = 0.1 * nq
    else:
        sec_per_eval = 60.0
    total = sec_per_eval * (2 * nq * 3 + 1) * maxiter * 2
    total = max(300, min(int(total), 48 * 3600))
    h, m = divmod(total // 60, 60)
    return f"{h:02d}:{m:02d}:00"
