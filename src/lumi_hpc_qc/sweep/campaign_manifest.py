# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Campaign manifest — tracks task completion across QPU batch submissions.

If Q50 drops mid-sweep, the manifest records which tasks completed,
which failed, and which were never submitted. On resume, the sweep
engine reads the manifest and executes only pending tasks.

Task ID format: s{seed}_p{placement_id}_g{pauli_group_index}
  e.g. s0_p7_g0 = seed 0, placement 7, Z-basis group

Persistence: JSON file written atomically (temp file + os.rename) after
each batch returns. This ensures the manifest is never corrupted by a
crash during write.

RED-DIRECTIVE-V130-v1.0 §5 — Campaign Manifest (Mandatory)
ORANGE-TO-RED-COMMS-019 v2.1 §7 — Campaign reliability gap
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any


class TaskStatus(str, Enum):
    """Status of an individual task in the campaign."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class BatchRecord:
    """Record of a single QPU batch submission."""
    batch_id: str
    task_ids: list[str]
    status: str  # "completed" or "failed"
    n_circuits: int = 0
    submitted_at: str = ""
    completed_at: str = ""
    error: str | None = None
    attempt: int = 1  # retry attempt number (1-based)


@dataclass
class CampaignManifest:
    """Tracks task completion for a QPU campaign.

    Written atomically after each batch. On resume, pending_tasks()
    returns only tasks not yet completed.

    Usage:
        # Create at sweep start
        manifest = CampaignManifest.create(sweep_id, task_ids)

        # After each batch
        manifest.mark_batch_completed(batch_id, task_ids)
        manifest.save(output_dir / "campaign_manifest.json")

        # On resume
        manifest = CampaignManifest.load(output_dir / "campaign_manifest.json")
        remaining = manifest.pending_tasks()
    """
    campaign_id: str
    total_tasks: int
    created_at: str = ""
    updated_at: str = ""
    framework_version: str = ""
    tasks: dict[str, str] = field(default_factory=dict)  # task_id → TaskStatus value
    batches: list[dict] = field(default_factory=list)  # serialized BatchRecords

    @classmethod
    def create(
        cls,
        campaign_id: str,
        task_ids: list[str],
        framework_version: str = "",
    ) -> CampaignManifest:
        """Create a new manifest with all tasks in PENDING state."""
        now = _utc_iso()
        tasks = {tid: TaskStatus.PENDING.value for tid in task_ids}
        return cls(
            campaign_id=campaign_id,
            total_tasks=len(task_ids),
            created_at=now,
            updated_at=now,
            framework_version=framework_version,
            tasks=tasks,
            batches=[],
        )

    def save(self, path: Path | str) -> None:
        """Atomic write: write to temp file in same directory, then rename.

        os.rename() is atomic on POSIX (Lustre, ext4, tmpfs). If the
        process crashes between write and rename, the old manifest is
        intact — the temp file is orphaned and harmless.
        """
        path = Path(path)
        self.updated_at = _utc_iso()
        data = asdict(self)

        # Write to temp file in the same directory (same filesystem for rename)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(path.parent),
            prefix=".manifest_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.rename(tmp_path, str(path))
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: Path | str) -> CampaignManifest:
        """Load existing manifest for resume."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        return cls(
            campaign_id=data["campaign_id"],
            total_tasks=data["total_tasks"],
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
            framework_version=data.get("framework_version", ""),
            tasks=data.get("tasks", {}),
            batches=data.get("batches", []),
        )

    def mark_batch_completed(
        self,
        batch_id: str,
        task_ids: list[str],
        n_circuits: int = 0,
        submitted_at: str = "",
        attempt: int = 1,
    ) -> None:
        """Mark all tasks in a batch as completed."""
        now = _utc_iso()
        for tid in task_ids:
            if tid in self.tasks:
                self.tasks[tid] = TaskStatus.COMPLETED.value

        self.batches.append(asdict(BatchRecord(
            batch_id=batch_id,
            task_ids=task_ids,
            status="completed",
            n_circuits=n_circuits,
            submitted_at=submitted_at or now,
            completed_at=now,
            attempt=attempt,
        )))

    def mark_batch_failed(
        self,
        batch_id: str,
        task_ids: list[str],
        error: str,
        n_circuits: int = 0,
        submitted_at: str = "",
        attempt: int = 1,
    ) -> None:
        """Mark all tasks in a batch as failed."""
        now = _utc_iso()
        for tid in task_ids:
            if tid in self.tasks:
                self.tasks[tid] = TaskStatus.FAILED.value

        self.batches.append(asdict(BatchRecord(
            batch_id=batch_id,
            task_ids=task_ids,
            status="failed",
            n_circuits=n_circuits,
            submitted_at=submitted_at or now,
            completed_at=now,
            error=error,
            attempt=attempt,
        )))

    def pending_tasks(self) -> list[str]:
        """Return task IDs where status is not completed.

        Includes both PENDING (never submitted) and FAILED (submitted
        but errored) tasks. Both need to be executed on resume.
        """
        return [
            tid for tid, status in self.tasks.items()
            if status != TaskStatus.COMPLETED.value
        ]

    def completed_tasks(self) -> list[str]:
        """Return task IDs that completed successfully."""
        return [
            tid for tid, status in self.tasks.items()
            if status == TaskStatus.COMPLETED.value
        ]

    def summary(self) -> dict[str, int]:
        """Count tasks by status."""
        counts: dict[str, int] = {}
        for status in self.tasks.values():
            counts[status] = counts.get(status, 0) + 1
        return counts


def _utc_iso() -> str:
    """UTC timestamp in ISO 8601 format with timezone."""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
