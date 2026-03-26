# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Resume runner — continues VQE from checkpoint inside container.

Used by Mode B controller when a previous child job failed/timed out.
Reads RESUME_CHECKPOINT env var for the checkpoint file path.
"""

import os
import sys

project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(project_dir, "src"))
os.chdir(project_dir)

config_path = os.environ.get("CONFIG_PATH", "")
checkpoint_path = os.environ.get("RESUME_CHECKPOINT", "")

if not config_path:
    print("ERROR: CONFIG_PATH not set")
    sys.exit(1)
if not checkpoint_path:
    print("ERROR: RESUME_CHECKPOINT not set")
    sys.exit(1)

from lumi_hpc_qc.cli.config_loader import load_config
from lumi_hpc_qc.orchestration.controller import Controller

config = load_config(config_path)
print(f"Resuming experiment: {config.experiment_id}")
print(f"Checkpoint: {checkpoint_path}")

controller = Controller()
record = controller.resume(checkpoint_path, config)

print("")
print("=" * 60)
if record.convergence:
    err = record.convergence.relative_error_pct
    status = "PASSED" if (err is not None and err < 10.0) else "RESULT"
    print(f"  {status}  (error: {err:.4f}%)" if err is not None else f"  {status}")
    print(f"  Best energy:  {record.convergence.best_energy:.8f}")
    print(f"  Iterations:   {record.convergence.total_iterations}")
print("=" * 60)
