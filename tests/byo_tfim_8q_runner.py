# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""BYO TFIM 8q acceptance test — executed inside the container via GPU wrapper.

Expected result: E ≈ -7.40 (exact: -7.641), ~3% error.
This matches the lumi_vqa proof-of-concept result.
"""

import os
import sys

project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(project_dir, "src"))
os.chdir(project_dir)

from lumi_hpc_qc.cli.config_loader import load_config
from lumi_hpc_qc.orchestration.controller import Controller

config_path = os.path.join(project_dir, "configs", "byo_tfim_8q.yaml")
config = load_config(config_path)
print(f"Experiment ID: {config.experiment_id}")

controller = Controller()
record = controller.run_interactive(config)

print("")
print("=" * 60)
if record.convergence:
    err = record.convergence.relative_error_pct
    if err is not None and err < 10.0:
        print(f"  ACCEPTANCE TEST PASSED  (error: {err:.2f}%)")
    else:
        print(f"  ACCEPTANCE TEST FAILED  (error: {err}%)")
    print(f"  Best energy:  {record.convergence.best_energy:.8f}")
    print(f"  Exact energy: {record.convergence.exact_ground_energy}")
print("=" * 60)
