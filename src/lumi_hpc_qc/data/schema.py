# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""
data/schema.py — ExperimentRecord validation, schema v2.0.0

Keeps things simple: validation is plain Python (readable by anyone),
JSON Schema dict is provided as machine-readable documentation and used
when jsonschema is available (container has 4.26.0).

Schema version history:
  1.0.0  v1.0.0b1-b6  Implicit. Fields: experiment_id, config, provenance,
                       iterations, convergence, timing, circuit_metrics,
                       noise_config.
  2.0.0  v1.0.0b7     Adds: schema_version, error_mitigation_applied,
                       per_placement_results, noiseless_tier, quality_report.

Usage:
    from lumi_hpc_qc.data.schema import validate_record, SCHEMA_VERSION

    errors = validate_record(record_dict)    # returns list of strings
    if errors:
        print("\n".join(errors))            # empty = valid
"""

from __future__ import annotations

import math
from typing import Any

SCHEMA_VERSION = "2.0.0"

# Fields that must be present and non-null in every v2 record
_REQUIRED = [
    "experiment_id",
    "schema_version",
    "convergence",
    "iterations",
    "quality_report",
]

# noiseless_tier must be one of these values (or absent/null)
_VALID_TIERS = {1, 2, 3, None}


# ---------------------------------------------------------------------------
# Primary validation function — plain Python, no dependencies
# ---------------------------------------------------------------------------

def validate_record(record: dict) -> list[str]:
    """Validate a serialised ExperimentRecord dict against schema v2.0.0.

    Returns a list of human-readable error strings.
    An empty list means the record is valid.

    This is intentionally simple: plain Python checks, one per rule,
    easy to extend without knowing JSON Schema.

    Example::

        errors = validate_record(data)
        if errors:
            raise ValueError("Record failed validation:\n" + "\n".join(errors))
    """
    errors = []

    # 1. Required fields present and non-null
    for field in _REQUIRED:
        if field not in record:
            errors.append(f"missing required field: '{field}'")
        elif record[field] is None:
            errors.append(f"required field '{field}' is None")

    # Stop here if fundamentals are missing — later checks would be noisy
    if errors:
        return errors

    # 2. schema_version must be exactly "2.0.0"
    if record["schema_version"] != SCHEMA_VERSION:
        errors.append(
            f"schema_version is '{record['schema_version']}', expected '{SCHEMA_VERSION}'"
        )

    # 3. experiment_id must be a non-empty string
    if not isinstance(record["experiment_id"], str) or not record["experiment_id"]:
        errors.append("experiment_id must be a non-empty string")

    # 4. convergence must be a dict with total_iterations and best_energy
    conv = record.get("convergence")
    if not isinstance(conv, dict):
        errors.append("convergence must be a dict")
    else:
        if "total_iterations" not in conv:
            errors.append("convergence.total_iterations is missing")
        elif not isinstance(conv["total_iterations"], int) or conv["total_iterations"] < 0:
            errors.append("convergence.total_iterations must be a non-negative integer")

        if "best_energy" not in conv:
            errors.append("convergence.best_energy is missing")
        elif not isinstance(conv["best_energy"], (int, float)):
            errors.append("convergence.best_energy must be a number")
        elif math.isnan(conv["best_energy"]) or math.isinf(conv["best_energy"]):
            errors.append(f"convergence.best_energy is {conv['best_energy']} (invalid)")

    # 5. iterations must be a list
    if not isinstance(record.get("iterations"), list):
        errors.append("iterations must be a list")

    # 6. noiseless_tier must be 1, 2, 3, or null/absent
    if "noiseless_tier" in record and record["noiseless_tier"] not in _VALID_TIERS:
        errors.append(
            f"noiseless_tier must be 1, 2, 3, or null — got {record['noiseless_tier']!r}"
        )

    # 7. quality_report must be a dict with 'passed' bool and 'checks' dict
    qr = record.get("quality_report")
    if not isinstance(qr, dict):
        errors.append("quality_report must be a dict")
    else:
        if "passed" not in qr or not isinstance(qr["passed"], bool):
            errors.append("quality_report.passed must be a boolean")
        if "checks" not in qr or not isinstance(qr["checks"], dict):
            errors.append("quality_report.checks must be a dict")

    # 8. error_mitigation_applied: if present and not null, must be a dict
    ema = record.get("error_mitigation_applied")
    if ema is not None and not isinstance(ema, dict):
        errors.append("error_mitigation_applied must be a dict or null")

    return errors


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def is_valid(record: dict) -> bool:
    """Return True if the record passes all v2 validation checks."""
    return len(validate_record(record)) == 0


def is_v1_record(record: dict) -> bool:
    """Return True if this is a pre-Phase-D record (no schema_version field).

    v1 records were produced by v1.0.0b1-b6. They have the same nested
    structure but lack schema_version and the Phase D additions.
    """
    return "schema_version" not in record


def upgrade_v1_to_v2(record: dict) -> dict:
    """Add Phase D fields to a v1 record so it passes v2 validation.

    Returns a new dict — the original is never modified.
    Scientific values (energies, trajectories, configs) are untouched.
    """
    upgraded = dict(record)
    upgraded.setdefault("schema_version", SCHEMA_VERSION)
    upgraded.setdefault("error_mitigation_applied", None)
    upgraded.setdefault("per_placement_results", None)
    upgraded.setdefault("noiseless_tier", None)
    upgraded.setdefault("quality_report", {
        "passed": False,
        "checks": {},
        "warnings": ["Upgraded from v1 schema — quality gate not run"],
    })
    return upgraded


# ---------------------------------------------------------------------------
# JSON Schema dict — machine-readable documentation only
# Used when jsonschema library is available (LUMI container has 4.26.0).
# You do NOT need to understand JSON Schema to use this module.
# The validate_record() function above is the authoritative check.
# ---------------------------------------------------------------------------

EXPERIMENT_SCHEMA_V2: dict = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "ExperimentRecord v2.0.0",
    "description": "lumi-hpc-qc experiment record. Produced by ExperimentTracker.finalize().",
    "type": "object",
    "required": _REQUIRED,
    "additionalProperties": True,
    "properties": {
        "schema_version":           {"type": "string", "const": SCHEMA_VERSION},
        "experiment_id":            {"type": "string", "minLength": 1},
        "created_at":               {"type": "string"},
        "config":                   {"type": ["object", "null"]},
        "provenance":               {"type": ["object", "null"]},
        "convergence": {
            "type": "object",
            "required": ["total_iterations", "best_energy"],
            "properties": {
                "total_iterations":    {"type": "integer", "minimum": 0},
                "best_energy":         {"type": "number"},
                "exact_ground_energy": {"type": ["number", "null"]},
                "relative_error_pct":  {"type": ["number", "null"]},
                "optimizer_converged": {"type": "boolean"},
            },
        },
        "iterations": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["iteration", "energy"],
                "properties": {
                    "iteration":     {"type": "integer"},
                    "energy":        {"type": "number"},
                    "gradient_norm": {"type": ["number", "null"]},
                },
            },
        },
        "circuit_metrics":          {"type": ["object", "null"]},
        "noise_config":             {"type": ["object", "null"]},
        "error_mitigation_applied": {"type": ["object", "null"]},
        "per_placement_results":    {"type": ["array", "null"]},
        "noiseless_tier":           {"enum": [1, 2, 3, None]},
        "quality_report": {
            "type": "object",
            "required": ["passed", "checks"],
            "properties": {
                "passed":   {"type": "boolean"},
                "checks":   {"type": "object"},
                "warnings": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
}


def validate_with_jsonschema(record: dict) -> list[str]:
    """Run jsonschema validation — only available in the LUMI container.

    Falls back to validate_record() automatically if jsonschema is not installed.
    Prefer validate_record() unless you specifically need JSON Schema error paths.
    """
    try:
        from jsonschema import Draft7Validator
        validator = Draft7Validator(EXPERIMENT_SCHEMA_V2)
        return [
            f"[{' -> '.join(str(p) for p in e.absolute_path) or 'root'}] {e.message}"
            for e in sorted(validator.iter_errors(record),
                            key=lambda e: list(e.absolute_path))
        ]
    except ImportError:
        return validate_record(record)
