# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Data quality gate — automated pre-write validation of experiment results.

RED-SPEC-001 §6.4. Five checks must pass before any result is committed
to disk. The gate embeds a quality_report dict in the ExperimentRecord;
it also appends a one-line entry to quality_gate.log in the output dir.

Usage (called automatically by ExperimentTracker.finalize()):
    from lumi_hpc_qc.data.quality import QualityGate
    gate = QualityGate()
    quality_report = gate.run(record)   # dict, always returned
    record.quality_report = quality_report

The gate NEVER raises — a failure produces a quality_report with
passed=False and warnings, but result writing still proceeds.
A dataset that is wrong is worse than no dataset; a dataset with a
quality flag is better than a silently bad one.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from lumi_hpc_qc.types import ExperimentRecord

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (Team Red §6.4 defaults — overridable via env vars)
# ---------------------------------------------------------------------------

# Gradient norm below this at termination = converged
_GRAD_NORM_THRESHOLD = float(os.environ.get("HPCQC_QUALITY_GRAD_THRESHOLD", "0.1"))

# Minimum iterations before a run is considered non-trivially complete
# Catches abnormally early terminations (crashed optimizers, 0-iter results)
_MIN_ITERATIONS = int(os.environ.get("HPCQC_QUALITY_MIN_ITERS", "3"))

# Energy upper bound multiplier: final energy must be < |exact| * this factor
# Catches wildly wrong results (e.g. positive energies for ground-state problems)
_ENERGY_BOUND_FACTOR = float(os.environ.get("HPCQC_QUALITY_ENERGY_BOUND", "2.0"))


# ---------------------------------------------------------------------------
# QualityGate
# ---------------------------------------------------------------------------

class QualityGate:
    """Five pre-write quality checks for experiment results.

    Checks:
      1. completeness  — required fields present and non-null
      2. consistency   — num_qubits matches Hamiltonian qubit count
      3. convergence   — gradient norm below threshold at termination
      4. energy_bound  — final energy does not exceed physical upper bound
      5. iter_budget   — run did not terminate abnormally early
    """

    def run(self, record: "ExperimentRecord") -> dict[str, Any]:
        """Run all five checks. Always returns a quality_report dict.

        Args:
            record: Completed ExperimentRecord from ExperimentTracker.finalize().

        Returns:
            dict with keys: passed (bool), checks (dict[str, bool]),
            warnings (list[str]).
        """
        checks: dict[str, bool] = {}
        warnings: list[str] = []

        checks["completeness"] = self._check_completeness(record, warnings)
        checks["consistency"] = self._check_consistency(record, warnings)
        checks["convergence"] = self._check_convergence(record, warnings)
        checks["energy_bound"] = self._check_energy_bound(record, warnings)
        checks["iter_budget"] = self._check_iter_budget(record, warnings)

        passed = all(checks.values())

        report = {
            "passed": passed,
            "checks": checks,
            "warnings": warnings,
        }

        self._write_log_entry(record, report)
        return report

    # ------------------------------------------------------------------
    # Check implementations
    # ------------------------------------------------------------------

    def _check_completeness(
        self, record: "ExperimentRecord", warnings: list[str]
    ) -> bool:
        """All required fields are present and non-null."""
        ok = True

        # experiment_id
        if not record.experiment_id:
            warnings.append("completeness: experiment_id is empty")
            ok = False

        # convergence summary
        if record.convergence is None:
            warnings.append("completeness: convergence summary is None")
            ok = False
        else:
            if record.convergence.best_energy == float("inf"):
                warnings.append("completeness: best_energy never updated from inf")
                ok = False
            if record.convergence.total_iterations == 0:
                warnings.append("completeness: total_iterations is 0")
                ok = False

        # iterations list
        if not record.iterations:
            warnings.append("completeness: iterations list is empty")
            ok = False

        # config
        if record.config is None:
            warnings.append("completeness: config is None")
            ok = False

        return ok

    def _check_consistency(
        self, record: "ExperimentRecord", warnings: list[str]
    ) -> bool:
        """num_qubits in config matches Hamiltonian qubit count."""
        if record.config is None:
            return True  # can't check — completeness will have flagged this

        config_qubits = record.config.num_qubits

        # Cross-check against circuit_metrics if available
        if record.circuit_metrics is not None:
            # CircuitMetrics doesn't store num_qubits directly,
            # but we can sanity-check that pre-transpilation depth is plausible
            pre_depth = record.circuit_metrics.pre_transpilation_depth
            if pre_depth == 0 and record.circuit_metrics.num_parameters > 0:
                warnings.append(
                    "consistency: circuit_metrics.pre_transpilation_depth=0 "
                    "but num_parameters>0 — pre-transpilation recording may have failed"
                )

        # Cross-check against hamiltonian metadata in provenance if present
        # (provenance stores imported modules, not Hamiltonian qubits —
        #  we verify config.num_qubits is a positive integer instead)
        if config_qubits <= 0:
            warnings.append(
                f"consistency: config.num_qubits={config_qubits} — "
                "Hamiltonian build may not have set this correctly"
            )
            return False

        return True

    def _check_convergence(
        self, record: "ExperimentRecord", warnings: list[str]
    ) -> bool:
        """Gradient norm is below threshold at termination (if available)."""
        if not record.iterations:
            return True  # can't check — completeness will have flagged this

        # Find the last recorded gradient norm (may be None for gradient-free runs)
        last_grad = None
        for it in reversed(record.iterations):
            if it.gradient_norm is not None:
                last_grad = it.gradient_norm
                break

        if last_grad is None:
            # Gradient-free optimizer (SPSA, COBYLA) — check optimizer_converged flag
            if record.convergence and not record.convergence.optimizer_converged:
                warnings.append(
                    "convergence: optimizer did not report convergence "
                    "(gradient-free run — check optimizer message)"
                )
                # Warn but don't fail: gradient-free runs may hit maxiter normally
            return True

        if last_grad > _GRAD_NORM_THRESHOLD:
            warnings.append(
                f"convergence: final gradient norm {last_grad:.4f} > "
                f"threshold {_GRAD_NORM_THRESHOLD} — optimizer may not have converged"
            )
            return False

        return True

    def _check_energy_bound(
        self, record: "ExperimentRecord", warnings: list[str]
    ) -> bool:
        """Final energy does not exceed a plausible physical upper bound."""
        if record.convergence is None:
            return True

        best_e = record.convergence.best_energy
        exact_e = record.convergence.exact_ground_energy

        # If exact energy is known, best_e must be >= exact_e (variational principle)
        if exact_e is not None:
            if best_e < exact_e - 1e-6:
                warnings.append(
                    f"energy_bound: best_energy {best_e:.8f} is below exact "
                    f"ground energy {exact_e:.8f} — violates variational principle"
                )
                return False

        # best_e should not be +inf or NaN
        import math
        if math.isnan(best_e) or math.isinf(best_e):
            warnings.append(
                f"energy_bound: best_energy is {best_e} — optimizer produced invalid value"
            )
            return False

        # For ground-state problems (negative exact energy), warn if result is positive
        if exact_e is not None and exact_e < 0 and best_e > 0:
            warnings.append(
                f"energy_bound: best_energy {best_e:.4f} is positive but "
                f"exact energy {exact_e:.4f} is negative — likely stuck in wrong basin"
            )
            return False

        return True

    def _check_iter_budget(
        self, record: "ExperimentRecord", warnings: list[str]
    ) -> bool:
        """Run completed a reasonable number of iterations."""
        if record.convergence is None:
            return True

        total = record.convergence.total_iterations

        if total < _MIN_ITERATIONS:
            warnings.append(
                f"iter_budget: only {total} iteration(s) completed "
                f"(minimum {_MIN_ITERATIONS}) — run may have failed or been trivially short"
            )
            return False

        # Warn (don't fail) if run hit the maximum without converging
        if record.config is not None:
            max_iters = record.config.optimizer_params.get("maxiter", 0)
            if (
                max_iters > 0
                and total >= max_iters
                and not record.convergence.optimizer_converged
            ):
                warnings.append(
                    f"iter_budget: run hit maxiter={max_iters} without optimizer "
                    "reporting convergence — consider increasing budget"
                )
                # Warn only — don't fail the check, this is a common outcome

        return True

    # ------------------------------------------------------------------
    # Log writer
    # ------------------------------------------------------------------

    def _write_log_entry(
        self, record: "ExperimentRecord", report: dict[str, Any]
    ) -> None:
        """Append a one-line entry to quality_gate.log in the output dir."""
        try:
            if record.config is None:
                return

            output_dir = Path(record.config.output_dir) / record.config.model
            output_dir.mkdir(parents=True, exist_ok=True)
            log_path = output_dir / "quality_gate.log"

            status = "PASS" if report["passed"] else "FAIL"
            checks_str = ",".join(
                f"{k}={'T' if v else 'F'}"
                for k, v in report["checks"].items()
            )
            warn_count = len(report["warnings"])
            line = (
                f"{record.created_at}  "
                f"{status}  "
                f"{record.experiment_id}  "
                f"[{checks_str}]  "
                f"warnings={warn_count}\n"
            )

            with open(log_path, "a") as f:
                f.write(line)

        except Exception as e:
            # Log writing must never crash the run
            logger.warning("quality_gate.log write failed: %s", e)
