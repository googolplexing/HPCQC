# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""D3.3 — route `device_calibrated` through the noise `source` axis.

Verifies the routing-layer seam (config vocabulary + fail-loud), NOT execution:
  - NoiseConfig gains source="channels" (default) | "device_calibrated"
  - the 11 synthetic channel tiers are unchanged and stay source="channels"
  - device_calibrated is registered (resolvable by name, passes F-6 validation)
  - device_calibrated pins method="statevector" (validation rejects density_matrix)
  - "all" expands to the channel tiers only -> device_calibrated is opt-in by name
  - executing a device_calibrated env raises NotImplementedError (BYO path = D3.4)

Pure-Python (no qiskit): imports noise_configs by file path to avoid the
qiskit-importing package __init__. The execution-guard test asserts on the
source-set logic without running a sweep.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_NC_PATH = (
    Path(__file__).resolve().parents[2]
    / "src" / "lumi_hpc_qc" / "sweep" / "noise_configs.py"
)


def _load_noise_configs():
    spec = importlib.util.spec_from_file_location("nc_d33", _NC_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nc_d33"] = mod  # needed for frozen-dataclass type resolution
    spec.loader.exec_module(mod)
    return mod


nc = _load_noise_configs()
NoiseConfig = nc.NoiseConfig
NOISE_ENVIRONMENTS = nc.NOISE_ENVIRONMENTS
NOISE_ENV_BY_NAME = nc.NOISE_ENV_BY_NAME

_HISTORICAL_11 = {
    "noiseless", "topology_noiseless", "noise_1q_only", "noise_2q_only",
    "noise_t1_only", "noise_t2_only", "noise_readout_only", "noise_coherence",
    "noise_gates", "noise_gates_readout", "noise_full",
}


# ----------------------- vocabulary / registration -----------------------

def test_eleven_channel_tiers_preserved():
    """The 11 historical environments are intact and all source='channels'."""
    chan = [e for e in NOISE_ENVIRONMENTS if e.source == "channels"]
    assert {e.name for e in chan} == _HISTORICAL_11
    assert len(chan) == 11
    assert all(e.source == "channels" for e in chan)


def test_existing_env_defaults_unchanged():
    """A spot-check that an existing env's fields are byte-identical."""
    nf = NOISE_ENV_BY_NAME["noise_full"]
    assert nf.method == "density_matrix"
    assert nf.source == "channels"
    assert nf.tier == "full"
    assert nf.channels is not None and all(nf.channels.values())


def test_device_calibrated_registered():
    """device_calibrated is resolvable by name (so F-6 validation accepts it)."""
    assert "device_calibrated" in NOISE_ENV_BY_NAME
    dc = NOISE_ENV_BY_NAME["device_calibrated"]
    assert dc.source == "device_calibrated"
    assert dc.method == "statevector"
    assert dc.channels is None


def test_total_env_count():
    assert len(NOISE_ENVIRONMENTS) == 12  # 11 channel + 1 device


# ----------------------------- validation --------------------------------

def test_device_calibrated_rejects_density_matrix():
    """Setting method=density_matrix on a device_calibrated config fails loud."""
    with pytest.raises(ValueError, match="pins method='statevector'"):
        NoiseConfig(name="x", source="device_calibrated", method="density_matrix")


def test_device_calibrated_accepts_statevector():
    NoiseConfig(name="x", source="device_calibrated", method="statevector")


def test_unknown_source_rejected():
    with pytest.raises(ValueError, match="must be 'channels'"):
        NoiseConfig(name="y", source="bogus")


def test_channels_source_default_unaffected():
    """Default source is 'channels'; method is free there (e.g. density_matrix)."""
    c = NoiseConfig(name="z")
    assert c.source == "channels"
    assert c.method == "density_matrix"


# --------------- "all" excludes device_calibrated (opt-in) ----------------

def test_all_expansion_is_channels_only():
    """Mirror the engine's 'all' resolution: channel tiers only, no device."""
    all_envs = [e for e in NOISE_ENVIRONMENTS if e.source == "channels"]
    names = {e.name for e in all_envs}
    assert "device_calibrated" not in names
    assert names == _HISTORICAL_11


# --------------- execution guard logic (no sweep run) ---------------------

def test_execution_guard_detects_device_source():
    """The _execute_group guard refuses any non-channels source. Reproduce its
    detection set on a mixed env list."""
    envs = [NOISE_ENV_BY_NAME["noise_full"], NOISE_ENV_BY_NAME["device_calibrated"]]
    bad = sorted({e.source for e in envs if e.source != "channels"})
    assert bad == ["device_calibrated"]

    clean = [NOISE_ENV_BY_NAME["noise_full"], NOISE_ENV_BY_NAME["noiseless"]]
    assert sorted({e.source for e in clean if e.source != "channels"}) == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
