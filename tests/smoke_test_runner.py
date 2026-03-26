# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Smoke test runner — executed inside the container via GPU wrapper."""

import os
import sys

project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(project_dir, "src"))

print("=== Import Test ===")
from lumi_hpc_qc import __version__
print(f"  Package version: {__version__}")

from lumi_hpc_qc.types import ExperimentConfig, BackendCapabilities
print("  types.py: OK")

from lumi_hpc_qc.backends.base import Backend
from lumi_hpc_qc.backends.registry import BackendRegistry
print("  backends: OK")

from lumi_hpc_qc.orchestration.workflow import Workflow, VQEWorkflow
from lumi_hpc_qc.orchestration.controller import Controller
from lumi_hpc_qc.orchestration.scheduler import SlurmScheduler
from lumi_hpc_qc.orchestration.checkpoint import CheckpointManager
print("  orchestration: OK")

from lumi_hpc_qc.data.experiment import ExperimentTracker
from lumi_hpc_qc.data.provenance import ProvenanceCollector
from lumi_hpc_qc.data.timing import TimingTracker
from lumi_hpc_qc.data.result_store import save_json, load_json
print("  data: OK")

from lumi_hpc_qc.cli.config_loader import load_config
print("  cli: OK")

print("")
print("=== Plugin Discovery ===")
from lumi_hpc_qc.plugins.registry import PluginRegistry
r = PluginRegistry()
r.discover()
for t in ["hamiltonians", "ansatze", "optimizers", "gradients", "initializers", "error_mitigation"]:
    plugins = r.list_available(t)
    print(f"  {t}: {plugins}")

print("")
print("=== Backend Discovery ===")
br = BackendRegistry()
br.discover()
print(f"  backends: {br.list_available()}")

print("")
print("=== Provenance Capture ===")
prov = ProvenanceCollector().capture()
print(f"  Python: {prov.python_version}")
print(f"  Node: {prov.lumi_node}")
print(f"  SLURM job: {prov.slurm_job_id}")
print(f"  Container: {prov.container_tag}")
print(f"  Imported modules: {len(prov.imported_modules)}")
for name in ["qiskit", "qiskit_aer", "numpy", "scipy"]:
    v = prov.imported_modules.get(name, "not found")
    print(f"    {name}: {v}")

print("")
print("=== Config Loading ===")
config_path = os.path.join(project_dir, "configs", "byo_tfim_8q.yaml")
config = load_config(config_path)
print(f"  Model: {config.model}")
print(f"  Ansatz: {config.ansatz}")
print(f"  Optimizer: {config.optimizer}")
print(f"  Backend: {config.backend}")
print(f"  Precision: {config.precision}")
print(f"  Experiment ID: {config.experiment_id}")

print("")
print("=" * 60)
print("  SMOKE TEST PASSED")
print("=" * 60)
