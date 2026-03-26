# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Test checkpoint/resume by running BYO TFIM with low maxiter, then resuming.

This test validates:
  1. Checkpoint files are created during optimization
  2. Resume loads the checkpoint and continues
  3. Final energy is at least as good as the checkpoint energy
"""

import os
import sys

project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(project_dir, "src"))
os.chdir(project_dir)

from lumi_hpc_qc.cli.config_loader import load_config
from lumi_hpc_qc.orchestration.controller import Controller
from lumi_hpc_qc.orchestration.checkpoint import CheckpointManager

# Phase 1: Run short optimization (50 iterations, checkpoint every 20)
print("=" * 60)
print("  Phase 1: Short run with checkpointing")
print("=" * 60)

config_path = os.path.join(project_dir, "configs", "byo_tfim_8q.yaml")
config = load_config(config_path, overrides={
    "output_dir": "results_checkpoint_test",
})
config.optimizer_params["maxiter"] = 50
config.checkpoint.interval = 20

exp_id = config.experiment_id
print(f"Experiment ID: {exp_id}")
print(f"Max iterations: 50, checkpoint every 20")

controller = Controller()
record1 = controller.run_interactive(config)

energy_phase1 = record1.convergence.best_energy
print(f"\nPhase 1 best energy: {energy_phase1:.8f}")

# Verify checkpoint exists
mgr = CheckpointManager(config.checkpoint.directory)
cp_path = mgr.exists(exp_id)
if cp_path is None:
    print("FAIL: No checkpoint file found!")
    sys.exit(1)
print(f"Checkpoint found: {cp_path}")

# List all checkpoints
all_cps = mgr.list_checkpoints(exp_id)
print(f"Total checkpoints: {len(all_cps)}")
for cp in all_cps:
    print(f"  {cp}")

# Phase 2: Resume from checkpoint with more iterations
print("\n" + "=" * 60)
print("  Phase 2: Resume from checkpoint")
print("=" * 60)

config2 = load_config(config_path, overrides={
    "output_dir": "results_checkpoint_test",
})
config2.optimizer_params["maxiter"] = 200  # more budget
config2.experiment_id = exp_id  # keep same ID for checkpoint lookup

record2 = controller.resume(cp_path, config2)

energy_phase2 = record2.convergence.best_energy
print(f"\nPhase 2 best energy: {energy_phase2:.8f}")
print(f"Phase 1 best energy: {energy_phase1:.8f}")

# Verify improvement (or at least no regression)
print("\n" + "=" * 60)
if energy_phase2 <= energy_phase1 + 1e-8:
    print(f"  CHECKPOINT/RESUME TEST PASSED")
    print(f"  Phase 1: {energy_phase1:+.8f}")
    print(f"  Phase 2: {energy_phase2:+.8f}")
    print(f"  Improvement: {energy_phase1 - energy_phase2:.8f}")
else:
    print(f"  CHECKPOINT/RESUME TEST FAILED")
    print(f"  Phase 2 energy ({energy_phase2}) worse than Phase 1 ({energy_phase1})")
print("=" * 60)
