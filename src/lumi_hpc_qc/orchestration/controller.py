# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Controller — runs on a SLURM-allocated node to orchestrate workflows.

Mode A (interactive): User gets an salloc/sbatch, runs workflow in-process.
    No child SLURM jobs — backend runs on this node's resources.

Mode B (automated): Single sbatch allocates a CPU-only controller node.
    Controller submits child SLURM jobs to GPU/QPU partitions, monitors
    them, and auto-restarts from checkpoint on failure.

    Controller lifecycle:
      1. Generate child VQE script from config + LUMI launch pattern
      2. Submit child to standard-g (or appropriate partition)
      3. Poll child status every 30s + watch progress files
      4. On child COMPLETED → collect results → done
      5. On child FAILED/TIMEOUT → find checkpoint → resubmit → goto 3
      6. Max retries (default 3) prevents infinite resubmission

    The controller runs on a CPU node (standard partition). It needs no
    GPU, no container — just Python 3 and sbatch/squeue. User can
    disconnect from LUMI; the controller job continues.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from lumi_hpc_qc.backends.registry import BackendRegistry
from lumi_hpc_qc.orchestration.scheduler import SlurmScheduler
from lumi_hpc_qc.plugins.registry import PluginRegistry
from lumi_hpc_qc.types import ExperimentConfig, ExperimentRecord

if TYPE_CHECKING:
    from lumi_hpc_qc.orchestration.workflow import Workflow


_WORKFLOW_CLASSES: dict[str, str] = {
    "vqe": "lumi_hpc_qc.orchestration.workflow.VQEWorkflow",
    "vqa": "lumi_hpc_qc.orchestration.workflow.VQAWorkflow",
    "circuit_submission": "lumi_hpc_qc.orchestration.workflow.CircuitSubmissionWorkflow",
}


def _resolve_workflow(name: str) -> type:
    import importlib
    if name not in _WORKFLOW_CLASSES:
        raise KeyError(f"Unknown workflow '{name}'. Available: {', '.join(_WORKFLOW_CLASSES)}")
    module_path, class_name = _WORKFLOW_CLASSES[name].rsplit(".", 1)
    return getattr(importlib.import_module(module_path), class_name)


class Controller:
    """Orchestrates workflow execution in Mode A or Mode B."""

    def __init__(self) -> None:
        self._plugin_registry = PluginRegistry()
        self._backend_registry = BackendRegistry()
        self._scheduler = SlurmScheduler()

    def _setup_registries(self) -> None:
        self._plugin_registry.discover()
        self._backend_registry.discover()

    def run_interactive(self, config: ExperimentConfig) -> ExperimentRecord:
        """Mode A: run everything in-process on current node."""
        self._setup_registries()

        errors = self._plugin_registry.validate_config(config)
        errors += self._backend_registry.validate_config(config.backend, config)
        if errors:
            raise ValueError("Configuration errors:\n" + "\n".join(f"  - {e}" for e in errors))

        workflow_name = config.model_params.get("workflow", "vqe")
        workflow = _resolve_workflow(workflow_name)()
        return workflow.run(config)

    def run_automated(self, config: ExperimentConfig) -> ExperimentRecord:
        """Mode B: orchestrate child SLURM jobs from controller node.

        Mode B is implemented as a pure bash controller script that runs
        on a minimal CPU allocation (small partition). It does not need
        a container, GPU, or Python libraries — just sbatch/squeue.

        To launch Mode B:
            ./tests/launch_mode_b.sh configs/your_experiment.yaml

        The bash controller (tests/mode_b_controller.sh) handles:
          1. Generating child VQE scripts matching the LUMI launch pattern
          2. Submitting child jobs to standard-g
          3. Polling status + reading progress files
          4. Auto-retry from checkpoint on FAILED/TIMEOUT
          5. Collecting results from shared filesystem

        This Python method is not used in production — it exists for
        programmatic access from notebooks or scripts that want to
        interact with Mode B via the Controller API.
        """
        raise NotImplementedError(
            "Mode B is implemented as a bash controller.\n"
            "Use: ./tests/launch_mode_b.sh configs/your_experiment.yaml\n"
            "Or: sbatch tests/mode_b_controller.sh  (with MODEB_* env vars)"
        )

    def resume(self, checkpoint_path: str, config: ExperimentConfig) -> ExperimentRecord:
        """Resume a previously interrupted workflow from checkpoint."""
        self._setup_registries()
        workflow_name = config.model_params.get("workflow", "vqe")
        workflow = _resolve_workflow(workflow_name)()
        return workflow.resume(checkpoint_path, config)
