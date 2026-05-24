# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Regression tests for RED-RESP §7.5 F-6: unknown noise-config names must
fail loud, never be silently dropped.

Before the fix, ``expand_grid`` resolved ``noise_configs`` with
``[NOISE_ENV_BY_NAME[n] for n in nc_names if n in NOISE_ENV_BY_NAME]`` — the
trailing filter dropped any unknown name (a typo, or a not-yet-existing name
like ``device_calibrated``) with no error, so the sweep quietly ran a
*different* set of environments than the config requested. That is a latent
data-integrity bug affecting every experiment type.

These tests pin the loud-failure behavior at both layers:
  * ``expand_grid`` raises ``ValueError`` (the resolution point; Red's required
    ``[noiseless, bogus]`` regression), for the list AND the scalar form.
  * ``validate_sweep_config`` reports the unknown name (clean pre-submit error),
    now including the scalar form the old list-only check missed.
"""
from __future__ import annotations

import pytest

from lumi_hpc_qc.sweep.sweep_engine import (
    expand_grid,
    parse_sweep_config,
    validate_sweep_config,
)


def _cfg(noise_configs):
    """Minimal characterization sweep config with the given noise_configs.

    The noise resolution is the first step inside expand_grid's per-experiment
    loop (before qubit_sizes/calibrations are touched), so a bad name raises
    regardless of the rest of the config.
    """
    return parse_sweep_config({
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "seeds": 1,
                "noise_configs": noise_configs,
            }],
            "calibrations": ["dummy_cal.json"],
        }
    })


def test_expand_grid_raises_on_unknown_list():
    """Red's required regression: [noiseless, bogus] must raise."""
    with pytest.raises(ValueError, match="Unknown noise config"):
        expand_grid(_cfg(["noiseless", "bogus"]))


def test_expand_grid_raises_on_unknown_scalar():
    """Scalar form was missed by the old list-only validate path."""
    with pytest.raises(ValueError, match="Unknown noise config"):
        expand_grid(_cfg("bogus"))


def test_validate_reports_unknown_scalar():
    """validate_sweep_config now flags the scalar form too (clean early error)."""
    errors = validate_sweep_config(_cfg("bogus"))
    assert any("unknown noise config" in e for e in errors), errors


def test_validate_reports_unknown_in_list():
    """No regression: the list form is still reported by validate."""
    errors = validate_sweep_config(_cfg(["noiseless", "bogus"]))
    assert any("unknown noise config" in e for e in errors), errors


def test_expand_grid_accepts_known_names():
    """No regression: valid names still expand to tasks carrying the envs."""
    tasks = expand_grid(_cfg(["noiseless", "noise_full"]))
    assert len(tasks) >= 1
    assert all(t.noise_configs for t in tasks)


def test_expand_grid_all_keyword_unaffected():
    """`noise_configs: all` still resolves to the full environment list."""
    tasks = expand_grid(_cfg("all"))
    assert len(tasks) >= 1
    assert all(t.noise_configs for t in tasks)
