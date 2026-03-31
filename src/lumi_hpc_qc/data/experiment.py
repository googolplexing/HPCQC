# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Experiment tracker — central logging for experiment lifecycle.

Creates a unique experiment ID (UUID + SLURM job ID), logs every workflow
step with timestamps, produces structured JSON output for reproducibility,
benchmarking, and AI/ML training data consumption.

Usage:
    tracker = ExperimentTracker(config)
    tracker.start()
    for each iteration:
        tracker.log_iteration(record)
    experiment = tracker.finalize(result, timing)
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lumi_hpc_qc.types import (
    ConvergenceSummary,
    ExperimentConfig,
    ExperimentRecord,
    IterationRecord,
    OptimizeResult,
    ProvenanceData,
    TimingBreakdown,
)


class ExperimentTracker:
    """Tracks the full lifecycle of an experiment."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._experiment_id = config.experiment_id
        self._iterations: list[IterationRecord] = []
        self._best_energy: float = float("inf")
        self._best_iteration: int = 0
        self._provenance: ProvenanceData | None = None
        self._output_dir = Path(config.output_dir) / config.model
        self._started = False
        # Phase B: circuit metrics and noise config
        self._circuit_metrics = None
        self._noise_config = None
        # Phase D: error mitigation record, placement results, sweep tier, quality report
        self._error_mitigation_applied: dict | None = None
        self._per_placement_results: list | None = None
        self._noiseless_tier: int | None = None
        self._quality_report: dict | None = None
        # V19: measurement stats sidecar
        self._measurement_stats_path: Path | None = None
        self._measurement_stats_count: int = 0

    @property
    def experiment_id(self) -> str:
        return self._experiment_id

    def start(self, provenance: ProvenanceData | None = None) -> str:
        """Begin tracking a new experiment.

        Creates output directory, writes initial config snapshot.

        Args:
            provenance: Environment metadata (captured by ProvenanceCollector).

        Returns:
            experiment_id
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._provenance = provenance
        self._started = True

        # V19: initialize measurement stats sidecar if capture enabled
        if self._config.capture_measurement_stats:
            self._measurement_stats_path = (
                self._output_dir / f"{self._experiment_id}_measurement_stats.jsonl"
            )

        # Write initial config snapshot (survives crashes)
        config_path = self._output_dir / f"{self._experiment_id}_config.json"
        with open(config_path, "w") as f:
            json.dump(self._serialize_config(), f, indent=2)

        return self._experiment_id

    def log_iteration(self, record: IterationRecord) -> None:
        """Record one optimization iteration.

        Updates best-energy tracking and writes incremental progress
        file that survives crashes.
        """
        if record.energy < self._best_energy:
            self._best_energy = record.energy
            self._best_iteration = record.iteration
            record.is_best = True

        self._iterations.append(record)

        # Write incremental progress (overwrite each time)
        if len(self._iterations) % 10 == 0:
            self._write_progress()

    @property
    def measurement_stats_path(self) -> Path | None:
        """Path to the measurement stats sidecar JSONL, or None if disabled."""
        return self._measurement_stats_path

    @property
    def measurement_stats_interval(self) -> int:
        """Capture every Nth evaluation."""
        return self._config.measurement_stats_interval

    def write_measurement_stats(
        self, eval_count: int, iteration: int, stats: list[dict]
    ) -> None:
        """Append measurement stats for one energy evaluation to the JSONL sidecar.

        Each dict in stats represents one Pauli group's raw counts.
        This is crash-safe: JSONL is append-only, at most one line lost.

        Args:
            eval_count: Sequential evaluation number (for interval tracking).
            iteration:  Current optimizer iteration.
            stats:      List of per-group dicts from the backend.
        """
        if self._measurement_stats_path is None:
            return
        # Check interval
        if eval_count % self._config.measurement_stats_interval != 0:
            return

        with open(self._measurement_stats_path, "a") as f:
            for group_stat in stats:
                record = {
                    "evaluation": eval_count,
                    "iteration": iteration,
                    **group_stat,
                }
                f.write(json.dumps(record, default=str) + "\n")
                self._measurement_stats_count += 1

    def finalize(
        self,
        result: OptimizeResult,
        timing: TimingBreakdown,
        exact_energy: float | None = None,
    ) -> ExperimentRecord:
        """Close experiment and write final structured JSON.

        Args:
            result: Optimizer output.
            timing: Phase-level timing breakdown.
            exact_energy: Reference exact ground state energy (if known).

        Returns:
            Complete ExperimentRecord.
        """
        # Compute convergence summary
        abs_error = None
        rel_error = None
        if exact_energy is not None:
            abs_error = abs(self._best_energy - exact_energy)
            if abs(exact_energy) > 1e-10:
                rel_error = abs_error / abs(exact_energy) * 100.0

        convergence = ConvergenceSummary(
            total_iterations=len(self._iterations),
            best_energy=self._best_energy,
            best_iteration=self._best_iteration,
            final_energy=result.fun if result else self._best_energy,
            exact_ground_energy=exact_energy,
            absolute_error=abs_error,
            relative_error_pct=rel_error,
            total_circuit_evaluations=result.nfev if result else 0,
            total_gradient_evaluations=0,  # filled by workflow
            optimizer_converged=result.success if result else False,
            optimizer_message=result.message if result else "",
        )

        record = ExperimentRecord(
            experiment_id=self._experiment_id,
            config=self._config,
            provenance=self._provenance,
            iterations=self._iterations,
            convergence=convergence,
            timing=timing,
            circuit_metrics=self._circuit_metrics,
            noise_config=self._noise_config,
            created_at=datetime.now(timezone.utc).isoformat(),
            # Phase D fields
            schema_version="2.0.0",
            error_mitigation_applied=self._error_mitigation_applied,
            per_placement_results=self._per_placement_results,
            noiseless_tier=self._noiseless_tier,
        )

        # Phase D: run quality gate before writing
        try:
            from lumi_hpc_qc.data.quality import QualityGate
            gate = QualityGate()
            record.quality_report = gate.run(record)
        except Exception as e:
            record.quality_report = {
                "passed": False,
                "checks": {},
                "warnings": [f"Quality gate error: {e}"],
            }

        # Write final result
        self._write_final(record)

        return record

    def _write_progress(self) -> None:
        """Write incremental progress file (survives crashes)."""
        path = self._output_dir / f"{self._experiment_id}_progress.json"
        progress = {
            "experiment_id": self._experiment_id,
            "iterations_completed": len(self._iterations),
            "best_energy": self._best_energy,
            "best_iteration": self._best_iteration,
            "last_energy": self._iterations[-1].energy if self._iterations else None,
        }
        with open(path, "w") as f:
            json.dump(progress, f, indent=2)

    def _write_final(self, record: ExperimentRecord) -> None:
        """Write complete experiment record as JSON."""
        path = self._output_dir / f"{self._experiment_id}_result.json"
        data = self._serialize_record(record)
        # V19: reference measurement stats sidecar if it exists
        if self._measurement_stats_path and self._measurement_stats_count > 0:
            data["measurement_stats_sidecar"] = str(self._measurement_stats_path.name)
            data["measurement_stats_entries"] = self._measurement_stats_count
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _serialize_config(self) -> dict[str, Any]:
        """Convert ExperimentConfig to JSON-safe dict."""
        from dataclasses import asdict
        return asdict(self._config)

    def _serialize_record(self, record: ExperimentRecord) -> dict[str, Any]:
        """Convert ExperimentRecord to JSON-safe dict."""
        from dataclasses import asdict
        import numpy as np

        def _convert(obj: Any) -> Any:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_convert(v) for v in obj]
            return obj

        raw = asdict(record)
        return _convert(raw)
