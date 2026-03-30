#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Generate benchmark configuration files from base models × mode templates.

RED-SPEC-001 §2.3: 13 modes × 7 models = 91 configurations.

Usage:
    # Generate all configs for one model:
    python scripts/generate_configs.py configs/q50bench_tfim_4q_noiseless.yaml --all-modes

    # Generate all configs for all models:
    python scripts/generate_configs.py --all

    # Generate a specific mode for one model:
    python scripts/generate_configs.py configs/q50bench_tfim_4q_noiseless.yaml --mode noise_t1_only

Output: configs/generated/q50bench_{model}_{mode}.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml


# ── Mode Templates ──
# Each mode defines: optimizer, gradient, backend_params overrides, description.
# The base config provides: model, model_params, ansatz, ansatz_params.

CALIBRATION_FILE = "examples/q50_calibration_20260326.json"

MODE_TEMPLATES = {
    "noiseless": {
        "description": "Aer GPU statevector — ideal simulation, no noise, full connectivity",
        "optimizer": "l_bfgs_b",
        "optimizer_params": {"maxiter": 400, "gtol": 1.0e-6},
        "gradient": "parameter_shift",
        "backend_params": {
            "method": "statevector",
        },
    },
    "controlled": {
        "description": "Aer GPU statevector — SPSA optimizer (noisy optimizer, no hardware noise)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "statevector",
        },
    },
    "topology_noiseless": {
        "description": "Aer GPU statevector — Q50 topology (SWAPs) but no noise",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "statevector",
            "coupling_map_source": "calibration",
            "coupling_map_file": CALIBRATION_FILE,
        },
    },
    "noise_1q_only": {
        "description": "Single-qubit depolarizing only (from RB data)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": CALIBRATION_FILE,
            "noise_channels": {
                "single_qubit_depolarizing": True,
            },
        },
    },
    "noise_2q_only": {
        "description": "Two-qubit depolarizing only (from CZ fidelity, Q50 topology)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": CALIBRATION_FILE,
            "coupling_map_source": "calibration",
            "noise_channels": {
                "two_qubit_depolarizing": True,
            },
        },
    },
    "noise_t1_only": {
        "description": "T1 amplitude damping only (idle-time decoherence)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": CALIBRATION_FILE,
            "noise_channels": {
                "t1_relaxation": True,
            },
        },
    },
    "noise_t2_only": {
        "description": "T2 dephasing only (idle-time decoherence)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": CALIBRATION_FILE,
            "noise_channels": {
                "t2_dephasing": True,
            },
        },
    },
    "noise_readout_only": {
        "description": "Readout error only (symmetric model from fidelity)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": CALIBRATION_FILE,
            "noise_channels": {
                "readout_error": True,
            },
        },
    },
    "noise_coherence": {
        "description": "T1 + T2 combined decoherence (Tier B: always co-occur)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": CALIBRATION_FILE,
            "noise_channels": {
                "t1_relaxation": True,
                "t2_dephasing": True,
            },
        },
    },
    "noise_gates": {
        "description": "1q + 2q depolarizing (Tier B: gate fidelity view, Q50 topology)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": CALIBRATION_FILE,
            "coupling_map_source": "calibration",
            "noise_channels": {
                "single_qubit_depolarizing": True,
                "two_qubit_depolarizing": True,
            },
        },
    },
    "noise_gates_readout": {
        "description": "1q + 2q depolarizing + readout (Tier B: RB + readout calibration)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": CALIBRATION_FILE,
            "coupling_map_source": "calibration",
            "noise_channels": {
                "single_qubit_depolarizing": True,
                "two_qubit_depolarizing": True,
                "readout_error": True,
            },
        },
    },
    "noise_full": {
        "description": "All noise channels active (Tier C: full Q50 noise model)",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 200},
        "gradient": "none",
        "backend_params": {
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": CALIBRATION_FILE,
            "coupling_map_source": "calibration",
        },
    },
    "qpu": {
        "description": "Real VTT Q50 quantum computer via FiQCI",
        "optimizer": "spsa",
        "optimizer_params": {"maxiter": 100},
        "gradient": "none",
        "backend": "iqm_qpu",
        "backend_params": {
            "shots": 4096,
        },
    },
}

# Models and their base config files
BASE_MODELS = {
    "tfim_2q": "configs/q50bench_tfim_2q_noiseless.yaml",
    "tfim_4q": "configs/q50bench_tfim_4q_noiseless.yaml",
    "tfim_8q": "configs/q50bench_tfim_8q_noiseless.yaml",
    "h2_4q": "configs/q50bench_h2_4q_noiseless.yaml",
    "qaoa_8q": "configs/q50bench_qaoa_8q_noiseless.yaml",
    "fh_4q": "configs/q50bench_fh_4q_noiseless.yaml",
    "heis_4q": "configs/q50bench_heis_4q_noiseless.yaml",
}


def load_base_config(path: str) -> dict:
    """Load a base model config, extracting model-specific fields."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def generate_config(base: dict, mode_name: str, model_name: str) -> dict:
    """Merge base model config with mode template."""
    template = MODE_TEMPLATES[mode_name]

    config = {
        "model": base["model"],
        "model_params": base.get("model_params", {}),
        "ansatz": base["ansatz"],
        "ansatz_params": base.get("ansatz_params", {}),
        "optimizer": template["optimizer"],
        "optimizer_params": template["optimizer_params"],
        "gradient": template["gradient"],
        "initializer": base.get("initializer", "random"),
        "initializer_params": base.get("initializer_params", {"seed": 42}),
        "backend": template.get("backend", "aer_gpu"),
        "backend_params": template["backend_params"],
        "precision": base.get("precision", "double"),
        "mode": "interactive",
        "checkpoint": {"enabled": True, "directory": "checkpoints", "interval": 50},
        "output_dir": "results",
    }
    return config


def write_config(config: dict, output_path: str, model_name: str, mode_name: str) -> None:
    """Write a generated config to YAML."""
    template = MODE_TEMPLATES[mode_name]
    header = (
        f"# Auto-generated by scripts/generate_configs.py\n"
        f"# Model: {model_name}, Mode: {mode_name}\n"
        f"# {template['description']}\n"
        f"#\n"
    )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(header)
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark configs")
    parser.add_argument("base_config", nargs="?", help="Path to base model config")
    parser.add_argument("--mode", help="Generate a specific mode")
    parser.add_argument("--all-modes", action="store_true", help="Generate all 13 modes")
    parser.add_argument("--all", action="store_true", help="Generate all models × all modes")
    parser.add_argument("--output-dir", default="configs/generated", help="Output directory")
    args = parser.parse_args()

    if args.all:
        total = 0
        for model_name, base_path in BASE_MODELS.items():
            if not os.path.exists(base_path):
                print(f"  SKIP: {model_name} — base config not found: {base_path}")
                continue
            base = load_base_config(base_path)
            for mode_name in MODE_TEMPLATES:
                config = generate_config(base, mode_name, model_name)
                out = os.path.join(args.output_dir, f"q50bench_{model_name}_{mode_name}.yaml")
                write_config(config, out, model_name, mode_name)
                total += 1
        print(f"Generated {total} configs in {args.output_dir}/")
        return

    if not args.base_config:
        parser.error("Provide a base config or use --all")

    base = load_base_config(args.base_config)
    # Infer model name from filename
    model_name = Path(args.base_config).stem.replace("q50bench_", "").replace("_noiseless", "")

    modes = list(MODE_TEMPLATES.keys()) if args.all_modes else [args.mode]
    if args.mode and args.mode not in MODE_TEMPLATES:
        parser.error(f"Unknown mode: {args.mode}. Available: {', '.join(MODE_TEMPLATES)}")

    for mode_name in modes:
        config = generate_config(base, mode_name, model_name)
        out = os.path.join(args.output_dir, f"q50bench_{model_name}_{mode_name}.yaml")
        write_config(config, out, model_name, mode_name)
        print(f"  {out}")

    print(f"Generated {len(modes)} config(s)")


if __name__ == "__main__":
    main()
