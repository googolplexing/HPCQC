# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Checkpoint manager — serialize/deserialize workflow state for crash recovery.

Saves full VQE state: parameters, best energy, iteration count, config hash.
On resume, restores to the exact point of interruption. Uses JSON with
numpy array serialization for human-readability and portability.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


class CheckpointManager:
    """Manages workflow state persistence for crash recovery."""

    def __init__(self, checkpoint_dir: str = "checkpoints") -> None:
        self._dir = Path(checkpoint_dir)

    def save(
        self,
        workflow_state: dict[str, Any],
        iteration: int,
        experiment_id: str,
    ) -> str:
        """Serialize workflow state to disk.

        Saves parameters, best energy, best params, iteration count,
        and any additional state the workflow provides.

        Returns path to the saved checkpoint file.
        """
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._dir / f"{experiment_id}_iter{iteration:06d}.json"

        serializable = _make_serializable(workflow_state)
        serializable["_checkpoint_meta"] = {
            "iteration": iteration,
            "experiment_id": experiment_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Write atomically (write to tmp, then rename)
        tmp_path = path.with_suffix(".tmp")
        with open(tmp_path, "w") as f:
            json.dump(serializable, f, indent=2)
        tmp_path.rename(path)

        return str(path)

    def load(self, checkpoint_path: str) -> dict[str, Any]:
        """Deserialize workflow state from disk.

        Restores numpy arrays from the JSON representation.
        """
        with open(checkpoint_path) as f:
            data = json.load(f)
        return _restore_arrays(data)

    def exists(self, experiment_id: str) -> str | None:
        """Find the latest checkpoint for an experiment.

        Returns path to most recent checkpoint, or None if none exist.
        """
        if not self._dir.exists():
            return None
        candidates = sorted(self._dir.glob(f"{experiment_id}_iter*.json"))
        return str(candidates[-1]) if candidates else None

    def list_checkpoints(self, experiment_id: str) -> list[str]:
        """List all checkpoint files for an experiment, oldest first."""
        if not self._dir.exists():
            return []
        return [str(p) for p in sorted(self._dir.glob(f"{experiment_id}_iter*.json"))]

    def cleanup(self, experiment_id: str, keep_latest: int = 3) -> int:
        """Remove old checkpoints, keeping the N most recent.

        Returns number of files deleted.
        """
        checkpoints = self.list_checkpoints(experiment_id)
        if len(checkpoints) <= keep_latest:
            return 0
        to_delete = checkpoints[:-keep_latest]
        for path in to_delete:
            os.remove(path)
        return len(to_delete)


def _make_serializable(obj: Any) -> Any:
    """Recursively convert numpy types to JSON-safe representations."""
    if isinstance(obj, np.ndarray):
        return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(v) for v in obj]
    return obj


def _restore_arrays(obj: Any) -> Any:
    """Recursively restore numpy arrays from JSON representation."""
    if isinstance(obj, dict):
        if obj.get("__ndarray__"):
            return np.array(obj["data"], dtype=obj.get("dtype", "float64"))
        return {k: _restore_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_restore_arrays(v) for v in obj]
    return obj
