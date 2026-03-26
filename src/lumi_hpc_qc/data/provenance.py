# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Provenance collector — captures reproducibility metadata.

Automatically records: Python version, all imported module versions,
container tag, hardware info, SLURM job details, git commit, and
QPU calibration data. Attached to every ExperimentRecord.

Container note: records the image tag/name from SINGULARITY_NAME env var,
NOT the hash — hashing an 11GB container is impractical.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import subprocess
import sys
from typing import Any

from lumi_hpc_qc.types import ProvenanceData


class ProvenanceCollector:
    """Captures all reproducibility metadata from the current environment."""

    def capture(self) -> ProvenanceData:
        """Collect provenance data from the runtime environment.

        Safe to call in any environment — gracefully handles missing
        tools (rocm-smi, nvidia-smi, git) by leaving fields empty.
        """
        return ProvenanceData(
            python_version=platform.python_version(),
            qiskit_version=self._get_package_version("qiskit"),
            qiskit_aer_version=self._get_package_version("qiskit-aer"),
            numpy_version=self._get_package_version("numpy"),
            scipy_version=self._get_package_version("scipy"),
            imported_modules=self._get_all_imported_modules(),
            container_tag=os.environ.get("SINGULARITY_NAME", ""),
            lumi_node=os.environ.get("SLURMD_NODENAME", platform.node()),
            gpu_model=self._detect_gpu_model(),
            gpu_memory_gb=self._detect_gpu_memory(),
            cpu_model=self._detect_cpu_model(),
            total_memory_gb=self._detect_total_memory(),
            slurm_job_id=os.environ.get("SLURM_JOB_ID", ""),
            slurm_partition=os.environ.get("SLURM_JOB_PARTITION", ""),
            slurm_num_nodes=int(os.environ.get("SLURM_JOB_NUM_NODES", "0")),
            git_commit=self._detect_git_commit(),
            git_branch=self._detect_git_branch(),
            git_dirty=self._detect_git_dirty(),
            q50_calibration=None,  # populated by IQM backend if used
        )

    @staticmethod
    def _get_package_version(package_name: str) -> str:
        """Get installed version of a Python package."""
        try:
            return importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            return "not installed"

    @staticmethod
    def _get_all_imported_modules() -> dict[str, str]:
        """Get versions of all currently imported Python modules.

        Walks sys.modules to find what was actually imported at runtime,
        then looks up their installed versions. This captures every
        dependency that contributed to the experiment results.
        """
        modules: dict[str, str] = {}
        seen_top_level: set[str] = set()

        for mod_name in sorted(sys.modules.keys()):
            # Only record top-level packages (not sub-modules)
            top = mod_name.split(".")[0]
            if top.startswith("_") or top in seen_top_level:
                continue
            seen_top_level.add(top)

            try:
                version = importlib.metadata.version(top)
                modules[top] = version
            except (importlib.metadata.PackageNotFoundError, ValueError):
                # stdlib modules and unpackaged modules don't have versions
                pass

        return modules

    @staticmethod
    def _detect_gpu_model() -> str | None:
        """Detect GPU model via rocm-smi (AMD/LUMI) or nvidia-smi."""
        # Try AMD first (LUMI uses MI250X)
        try:
            result = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.split("\n"):
                    if "GPU" in line or "gfx" in line or "MI" in line:
                        return line.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try NVIDIA (Roihu, future clusters)
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                return result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        return None

    @staticmethod
    def _detect_gpu_memory() -> float | None:
        """Detect GPU memory in GB."""
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            if result.returncode == 0 and result.stdout.strip():
                mb = float(result.stdout.strip().split("\n")[0])
                return round(mb / 1024.0, 1)
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
            pass
        return None

    @staticmethod
    def _detect_cpu_model() -> str:
        """Read CPU model from /proc/cpuinfo or platform."""
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except (FileNotFoundError, PermissionError):
            pass
        return platform.processor() or "unknown"

    @staticmethod
    def _detect_total_memory() -> float | None:
        """Detect total system memory in GB from /proc/meminfo."""
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(line.split()[1])
                        return round(kb / 1024 / 1024, 1)
        except (FileNotFoundError, PermissionError, ValueError):
            pass
        return None

    @staticmethod
    def _detect_git_commit() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _detect_git_branch() -> str | None:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            return result.stdout.strip() if result.returncode == 0 else None
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return None

    @staticmethod
    def _detect_git_dirty() -> bool:
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True, text=True, timeout=5, check=False,
            )
            return bool(result.stdout.strip()) if result.returncode == 0 else False
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
