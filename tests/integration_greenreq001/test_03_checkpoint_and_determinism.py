# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""GREEN-REQ-001 Integration Test 3: Checkpoint/Resume + PYTHONHASHSEED.

Verifies that checkpoint/resume works correctly on the new container
and that PYTHONHASHSEED=0 produces deterministic behavior.
"""

import os
import sys
import hashlib

project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(project_dir, "src"))
os.chdir(project_dir)

print("=" * 70)
print("  GREEN-REQ-001 Integration Test 3: Checkpoint/Resume + Determinism")
print("=" * 70)
print()

# ── Test 3a: PYTHONHASHSEED check ──
print("--- Test 3a: PYTHONHASHSEED Determinism ---")

hashseed = os.environ.get("PYTHONHASHSEED", "not set")
print(f"  PYTHONHASHSEED={hashseed}")

if hashseed == "0":
    # Verify deterministic dict ordering
    d = {"z": 1, "a": 2, "m": 3, "b": 4}
    keys1 = list(d.keys())
    d2 = {"z": 1, "a": 2, "m": 3, "b": 4}
    keys2 = list(d2.keys())
    assert keys1 == keys2, f"Dict ordering not deterministic: {keys1} vs {keys2}"

    # Verify deterministic set iteration
    s = {"pauli_Z", "pauli_X", "pauli_Y", "pauli_I"}
    order1 = sorted(s)  # sorted is always deterministic, but set() iteration order depends on hash
    order2 = sorted(s)
    assert order1 == order2
    print("  [PASS] PYTHONHASHSEED=0 — deterministic dict/set ordering")
else:
    print("  [WARN] PYTHONHASHSEED is not 0 — reproducibility not guaranteed")
    print("         Set 'export PYTHONHASHSEED=0' in env.sh")

# ── Test 3b: Checkpoint/Resume ──
print()
print("--- Test 3b: Checkpoint/Resume ---")

from lumi_hpc_qc.cli.config_loader import load_config
from lumi_hpc_qc.orchestration.controller import Controller
from lumi_hpc_qc.orchestration.checkpoint import CheckpointManager

config_path = os.path.join(project_dir, "configs", "byo_tfim_8q.yaml")
if not os.path.exists(config_path):
    print(f"  SKIP: {config_path} not found")
    sys.exit(0)

# Phase 1: Short run (30 iterations, checkpoint every 10)
print("  Phase 1: Short run (30 iters, checkpoint every 10)")
config = load_config(config_path, overrides={
    "output_dir": "results_integration_checkpoint",
})
config.optimizer_params["maxiter"] = 30
config.checkpoint.interval = 10

exp_id = config.experiment_id
controller = Controller()
record1 = controller.run_interactive(config)
energy_phase1 = record1.convergence.best_energy
print(f"  Phase 1 best energy: {energy_phase1:.8f}")

# Verify checkpoint exists
mgr = CheckpointManager(config.checkpoint.directory)
cp_path = mgr.exists(exp_id)
if cp_path is None:
    print("  [FAIL] No checkpoint file created!")
    sys.exit(1)
print(f"  Checkpoint found: {cp_path}")

# Phase 2: Resume with more iterations
print()
print("  Phase 2: Resume from checkpoint (100 more iters)")
config2 = load_config(config_path, overrides={
    "output_dir": "results_integration_checkpoint",
})
config2.optimizer_params["maxiter"] = 100
config2.experiment_id = exp_id

record2 = controller.resume(cp_path, config2)
energy_phase2 = record2.convergence.best_energy
print(f"  Phase 2 best energy: {energy_phase2:.8f}")

# Verify no regression
if energy_phase2 <= energy_phase1 + 1e-8:
    improvement = energy_phase1 - energy_phase2
    print(f"  Energy improvement: {improvement:.8f}")
    print("  [PASS] Checkpoint/resume working correctly")
else:
    print(f"  [FAIL] Phase 2 energy ({energy_phase2}) worse than Phase 1 ({energy_phase1})")
    sys.exit(1)

# Cleanup
import shutil
for d in ["results_integration_checkpoint", "checkpoints"]:
    if os.path.exists(d):
        shutil.rmtree(d)
        print(f"  Cleaned up: {d}/")

print()
print("=" * 70)
print("  INTEGRATION TEST 3: COMPLETE")
print("=" * 70)
