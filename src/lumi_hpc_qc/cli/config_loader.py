# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Config loader — YAML parsing, validation, and CLI override merging.

Reads experiment.yaml into a typed ExperimentConfig dataclass.
Supports CLI overrides (e.g., --model fermi_hubbard --precision single).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from lumi_hpc_qc.types import CheckpointConfig, ExperimentConfig, SlurmConfig


def load_config(
    yaml_path: str | Path,
    overrides: dict[str, Any] | None = None,
) -> ExperimentConfig:
    """Load experiment config from YAML file with optional overrides.

    Args:
        yaml_path: Path to experiment YAML config file.
        overrides: Dict of field_name → value overrides from CLI.

    Returns:
        Validated ExperimentConfig.

    Raises:
        FileNotFoundError: If yaml_path doesn't exist.
        ValueError: If config has invalid values.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config file must be a YAML mapping, got {type(raw)}")

    # Apply CLI overrides
    if overrides:
        raw.update(overrides)

    return _build_config(raw)


def _build_config(raw: dict[str, Any]) -> ExperimentConfig:
    """Convert raw YAML dict to typed ExperimentConfig."""

    # Extract nested configs
    slurm_raw = raw.pop("slurm", {})
    checkpoint_raw = raw.pop("checkpoint", {})
    data_raw = raw.pop("data", {})  # V19: optional data: section

    def _get_data_field(raw_dict, key, default):
        """Check top-level first, then data: section."""
        if key in raw_dict:
            return raw_dict[key]
        if isinstance(data_raw, dict) and key in data_raw:
            return data_raw[key]
        return default

    slurm = SlurmConfig(
        partition=slurm_raw.get("partition", "standard-g"),
        account=slurm_raw.get("account", ""),
        walltime=slurm_raw.get("walltime", "01:00:00"),
        nodes=slurm_raw.get("nodes", 1),
        gpus_per_node=slurm_raw.get("gpus_per_node", 0),
        container_path=slurm_raw.get("container_path", ""),
        extra_sbatch_flags=slurm_raw.get("extra_sbatch_flags", {}),
    )

    checkpoint = CheckpointConfig(
        enabled=checkpoint_raw.get("enabled", True),
        directory=checkpoint_raw.get("directory", "checkpoints"),
        interval=checkpoint_raw.get("interval", 10),
    )

    config = ExperimentConfig(
        model=raw.get("model", ""),
        model_params=raw.get("model_params", {}),
        ansatz=raw.get("ansatz", ""),
        ansatz_params=raw.get("ansatz_params", {}),
        optimizer=raw.get("optimizer", "l_bfgs_b"),
        optimizer_params=raw.get("optimizer_params", {}),
        gradient=raw.get("gradient", "parameter_shift"),
        gradient_params=raw.get("gradient_params", {}),
        initializer=raw.get("initializer", "random"),
        initializer_params=raw.get("initializer_params", {}),
        backend=raw.get("backend", "aer_gpu"),
        backend_params=raw.get("backend_params", {}),
        error_mitigation=raw.get("error_mitigation"),
        error_mitigation_params=raw.get("error_mitigation_params", {}),
        precision=raw.get("precision", "double"),
        mode=raw.get("mode", "interactive"),
        slurm=slurm,
        checkpoint=checkpoint,
        # V19: measurement stats capture (RED-SPEC-001 §5.3.3)
        # Supports both top-level and data: section YAML placement
        capture_measurement_stats=_get_data_field(
            raw, "capture_measurement_stats", False),
        measurement_stats_interval=_get_data_field(
            raw, "measurement_stats_interval", 10),
        output_dir=raw.get("output_dir", "results"),
    )

    # Basic validation
    errors = _validate(config)
    if errors:
        raise ValueError(
            "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    return config


def _validate(config: ExperimentConfig) -> list[str]:
    """Validate config fields. Returns list of errors."""
    errors: list[str] = []

    if not config.model:
        errors.append("'model' is required")
    if not config.ansatz:
        errors.append("'ansatz' is required")
    if config.precision not in ("double", "single"):
        errors.append(f"'precision' must be 'double' or 'single', got '{config.precision}'")
    if config.mode not in ("interactive", "automated"):
        errors.append(f"'mode' must be 'interactive' or 'automated', got '{config.mode}'")

    return errors
