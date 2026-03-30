#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Run a single VQE experiment from a config file.

Usage:
    python scripts/run_vqe.py configs/seed_sweep/tfim_2q/q50bench_tfim_2q_noiseless_seed0042.yaml
"""

import sys
import os

# Ensure project is on path
project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

from lumi_hpc_qc.orchestration.workflow import VQEWorkflow
from lumi_hpc_qc.cli.config_loader import load_config


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/run_vqe.py <config.yaml>")
        sys.exit(1)

    config_path = sys.argv[1]
    print(f"Loading config: {config_path}")

    config = load_config(config_path)
    print(f"Model: {config.model}, Seed: {config.initializer_params.get('seed', 'N/A')}")

    workflow = VQEWorkflow()
    record = workflow.run(config)

    print(f"Experiment: {record.experiment_id}")
    print(f"Final energy: {record.convergence.best_energy:.8f}")
    print(f"Iterations: {record.convergence.total_iterations}")


if __name__ == "__main__":
    main()
