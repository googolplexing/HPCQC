# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""SLURM job lifecycle management.

Wraps subprocess calls to SLURM CLI (sbatch, squeue, sacct, scancel).
Used by controller.py for Mode B child job management.

This module has NO imports from lumi_hpc_qc — it is a pure SLURM
interaction layer.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path


class SlurmScheduler:
    """Manages SLURM job submission, monitoring, and cancellation."""

    def submit(self, script_path: str, dependency: str | None = None) -> str:
        """Submit a SLURM batch script.

        Args:
            script_path: Path to the .sh batch script.
            dependency: Optional dependency spec (e.g., "afterok:12345").

        Returns:
            SLURM job ID as a string.
        """
        cmd = ["sbatch"]
        if dependency:
            cmd += [f"--dependency={dependency}"]
        cmd.append(script_path)

        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed: {result.stderr.strip()}")

        for word in result.stdout.strip().split():
            if word.isdigit():
                return word
        raise RuntimeError(f"Could not parse job ID from: {result.stdout}")

    def status(self, job_id: str) -> str:
        """Get job status via sacct.

        Returns: PENDING, RUNNING, COMPLETED, FAILED, TIMEOUT, CANCELLED, UNKNOWN.
        """
        # Try squeue first (faster, works for running/pending jobs)
        result = subprocess.run(
            ["squeue", "-j", job_id, "--format=%T", "--noheader"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            state = result.stdout.strip().split("\n")[0].strip()
            if state:
                return state.upper()

        # Fall back to sacct (works for completed jobs)
        result = subprocess.run(
            ["sacct", "-j", job_id, "--format=State", "--noheader", "--parsable2"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return "UNKNOWN"

        for line in result.stdout.strip().split("\n"):
            state = line.strip().split("|")[0] if "|" in line else line.strip()
            if state and state not in ("", "----------"):
                return state.upper()
        return "UNKNOWN"

    def wait(self, job_id: str, poll_interval_s: int = 30,
             progress_callback=None) -> str:
        """Block until job completes.

        Args:
            job_id: SLURM job ID.
            poll_interval_s: Seconds between status checks.
            progress_callback: Called each poll with (job_id, status, elapsed_s).

        Returns:
            Final status (COMPLETED, FAILED, TIMEOUT, CANCELLED).
        """
        terminal = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED"}
        start = time.time()
        while True:
            state = self.status(job_id)
            elapsed = time.time() - start
            if progress_callback:
                progress_callback(job_id, state, elapsed)
            if state in terminal:
                return state
            time.sleep(poll_interval_s)

    def cancel(self, job_id: str) -> None:
        """Cancel a running or pending job and all children."""
        subprocess.run(
            ["scancel", "--full", job_id],
            capture_output=True, text=True, check=False,
        )

    def job_info(self, job_id: str) -> dict:
        """Get detailed job info from sacct.

        Returns dict with: state, exit_code, elapsed, max_rss, node_list.
        """
        result = subprocess.run(
            ["sacct", "-j", job_id,
             "--format=State,ExitCode,Elapsed,MaxRSS,NodeList",
             "--noheader", "--parsable2"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0:
            return {"state": "UNKNOWN"}

        for line in result.stdout.strip().split("\n"):
            parts = line.strip().split("|")
            if len(parts) >= 5 and parts[0].strip():
                return {
                    "state": parts[0].strip(),
                    "exit_code": parts[1].strip(),
                    "elapsed": parts[2].strip(),
                    "max_rss": parts[3].strip(),
                    "node_list": parts[4].strip(),
                }
        return {"state": "UNKNOWN"}

    def collect_output(self, job_id: str, output_dir: str) -> dict | None:
        """Read JSON results from a completed job's output directory."""
        output_path = Path(output_dir)
        candidates = list(output_path.glob(f"*{job_id}*.json"))
        if not candidates:
            candidates = list(output_path.glob("*_result.json"))
        if not candidates:
            return None
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        with open(latest) as f:
            return json.load(f)
