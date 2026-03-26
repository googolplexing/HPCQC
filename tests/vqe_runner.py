# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Generic VQE experiment runner — executed inside container via GPU wrapper.

Reads config path from CONFIG_PATH environment variable.
Used by all model-specific sbatch scripts.
"""

import os
import sys

project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(project_dir, "src"))
os.chdir(project_dir)

config_path = os.environ.get("CONFIG_PATH", "")
if not config_path:
    print("ERROR: CONFIG_PATH environment variable not set")
    sys.exit(1)

from lumi_hpc_qc.cli.config_loader import load_config
from lumi_hpc_qc.orchestration.controller import Controller

config = load_config(config_path)
print(f"Experiment ID: {config.experiment_id}")
print(f"Config: {config_path}")
print("")

controller = Controller()
record = controller.run_interactive(config)

print("")
print("=" * 60)
if record.convergence:
    err = record.convergence.relative_error_pct
    status = "PASSED" if (err is not None and err < 10.0) else "RESULT"
    print(f"  {status}  (error: {err:.4f}%)" if err is not None else f"  {status}")
    print(f"  Best energy:  {record.convergence.best_energy:.8f}")
    if record.convergence.exact_ground_energy is not None:
        print(f"  Exact energy: {record.convergence.exact_ground_energy:.8f}")
    print(f"  Iterations:   {record.convergence.total_iterations}")
print("=" * 60)
