# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Noise environment configurations for the twin simulation battery.

Defines the 11 noise environments per placement per calibration.
Each environment specifies which noise channels are active, the
simulation method, shot count, and measurement stats capture interval.

RED-SPEC-002 §4.1 — The 11+1 Environments
RED-SPEC-002 §11.1 — Tiered Measurement Stats Capture
RED-DIRECTIVE-E4-SCHEMA-v1.0 §6.2 — Tiered intervals wired from start
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class NoiseConfig:
    """Configuration for one noise environment.

    Attributes:
        name: Canonical name (e.g., "noise_full", "noise_t1_only").
        channels: Dict of channel_name → bool. None = noiseless.
        method: Simulation method — "statevector" or "density_matrix".
        shots: Number of measurement shots (0 = exact).
        measurement_stats_interval: Capture interval (0 = disabled).
        coupling_map_source: "calibration" (Q50 topology) or "full" (all-to-all).
        tier: "A" (individual), "B" (combined), "full", or "noiseless".
        description: Human-readable description.
        source: noise vocabulary / execution path (D3). "channels" (default) =
            the synthetic depolarizing/T1/T2/readout channels run by the
            density_matrix twin battery (all 11 environments below).
            "device_calibrated" = real Q50-calibrated noise built by
            backends/device_noise via prepare_simulation under a pinned
            statevector method; executed by the BYO counts path (D3.4), NOT the
            twin battery. Defaulting to "channels" keeps the 11 environments
            byte-identical (RED-SPEC-002 preserved).
    """
    name: str
    channels: dict[str, bool] | None = None
    method: str = "density_matrix"
    shots: int = 4096
    measurement_stats_interval: int = 0
    coupling_map_source: str = "calibration"
    tier: str = "A"
    description: str = ""
    source: str = "channels"

    def __post_init__(self):
        # D3.3: device_calibrated runs through prepare_simulation, which PINS
        # method="statevector" (backends/prepare.py:410) -- that pin is the D2
        # correctness fix (deterministic Kraus precompute) AND the scalable
        # O(2^n) path, not a preference. A NoiseConfig.method on this source is
        # therefore not consultable; fail loud rather than silently ignore a
        # caller who set density_matrix expecting a different (state-level)
        # result. A density_matrix device-calibrated MODE is a separate future
        # increment (tracked as DEBT, not a knob on this config).
        if self.source not in ("channels", "device_calibrated"):
            raise ValueError(
                f"NoiseConfig.source must be 'channels' or 'device_calibrated', "
                f"got {self.source!r}"
            )
        if self.source == "device_calibrated" and self.method != "statevector":
            raise ValueError(
                f"device_calibrated noise pins method='statevector' "
                f"(the D2 Kraus-precompute fix + the scalable path); "
                f"got method={self.method!r}. density_matrix device-calibrated "
                f"simulation is a separate future mode, not a method override."
            )


# ═══════════════════════════════════════════════════════════════════════
# The 11 noise environments
# ═══════════════════════════════════════════════════════════════════════

# Channel shorthand
_NONE = None  # No noise model at all
_1Q   = {"single_qubit_depolarizing": True,  "two_qubit_depolarizing": False,
         "t1_relaxation": False, "t2_dephasing": False, "readout_error": False}
_2Q   = {"single_qubit_depolarizing": False, "two_qubit_depolarizing": True,
         "t1_relaxation": False, "t2_dephasing": False, "readout_error": False}
_T1   = {"single_qubit_depolarizing": False, "two_qubit_depolarizing": False,
         "t1_relaxation": True,  "t2_dephasing": False, "readout_error": False}
_T2   = {"single_qubit_depolarizing": False, "two_qubit_depolarizing": False,
         "t1_relaxation": False, "t2_dephasing": True,  "readout_error": False}
_RO   = {"single_qubit_depolarizing": False, "two_qubit_depolarizing": False,
         "t1_relaxation": False, "t2_dephasing": False, "readout_error": True}
_COH  = {"single_qubit_depolarizing": False, "two_qubit_depolarizing": False,
         "t1_relaxation": True,  "t2_dephasing": True,  "readout_error": False}
_GAT  = {"single_qubit_depolarizing": True,  "two_qubit_depolarizing": True,
         "t1_relaxation": False, "t2_dephasing": False, "readout_error": False}
_GRO  = {"single_qubit_depolarizing": True,  "two_qubit_depolarizing": True,
         "t1_relaxation": False, "t2_dephasing": False, "readout_error": True}
_ALL  = {"single_qubit_depolarizing": True,  "two_qubit_depolarizing": True,
         "t1_relaxation": True,  "t2_dephasing": True,  "readout_error": True}


NOISE_ENVIRONMENTS: list[NoiseConfig] = [
    # ── Noiseless (statevector, no shots, no stats) ──
    NoiseConfig(
        name="noiseless",
        channels=_NONE,
        method="statevector",
        shots=0,
        measurement_stats_interval=0,
        coupling_map_source="none",
        tier="noiseless",
        description="Ideal statevector — no noise, no routing, no shots",
    ),
    NoiseConfig(
        name="topology_noiseless",
        channels=_NONE,
        method="statevector",
        shots=0,
        measurement_stats_interval=0,
        coupling_map_source="calibration",
        tier="noiseless",
        description="Noiseless but transpiled to placement topology — routing overhead only",
    ),

    # ── Tier A: Individual noise channels (interval=5) ──
    NoiseConfig(
        name="noise_1q_only",
        channels=_1Q,
        method="density_matrix",
        shots=4096,
        measurement_stats_interval=5,
        coupling_map_source="full",
        tier="A",
        description="Single-qubit depolarizing only (from RB gate error)",
    ),
    NoiseConfig(
        name="noise_2q_only",
        channels=_2Q,
        method="density_matrix",
        shots=4096,
        measurement_stats_interval=5,
        coupling_map_source="calibration",
        tier="A",
        description="Two-qubit depolarizing only (from CZ fidelity)",
    ),
    NoiseConfig(
        name="noise_t1_only",
        channels=_T1,
        method="density_matrix",
        shots=4096,
        measurement_stats_interval=5,
        coupling_map_source="full",
        tier="A",
        description="T1 amplitude damping only",
    ),
    NoiseConfig(
        name="noise_t2_only",
        channels=_T2,
        method="density_matrix",
        shots=4096,
        measurement_stats_interval=5,
        coupling_map_source="full",
        tier="A",
        description="T2 dephasing only",
    ),
    NoiseConfig(
        name="noise_readout_only",
        channels=_RO,
        method="density_matrix",
        shots=4096,
        measurement_stats_interval=5,
        coupling_map_source="full",
        tier="A",
        description="Readout error only",
    ),

    # ── Tier B: Physically motivated pairs (interval=20) ──
    NoiseConfig(
        name="noise_coherence",
        channels=_COH,
        method="density_matrix",
        shots=4096,
        measurement_stats_interval=20,
        coupling_map_source="full",
        tier="B",
        description="T1 + T2 combined (decoherence)",
    ),
    NoiseConfig(
        name="noise_gates",
        channels=_GAT,
        method="density_matrix",
        shots=4096,
        measurement_stats_interval=20,
        coupling_map_source="calibration",
        tier="B",
        description="1q + 2q depolarizing (gate errors)",
    ),
    NoiseConfig(
        name="noise_gates_readout",
        channels=_GRO,
        method="density_matrix",
        shots=4096,
        measurement_stats_interval=20,
        coupling_map_source="calibration",
        tier="B",
        description="1q + 2q depol + readout (all non-coherence noise)",
    ),

    # ── Full noise (interval=10) ──
    NoiseConfig(
        name="noise_full",
        channels=_ALL,
        method="density_matrix",
        shots=4096,
        measurement_stats_interval=10,
        coupling_map_source="calibration",
        tier="full",
        description="All 5 noise channels active",
    ),

    # ── Device-calibrated (D3) — real Q50 noise, statevector counts path ──
    # source="device_calibrated" routes to backends/device_noise via
    # prepare_simulation (statevector pin), executed by the BYO counts path
    # (D3.4). channels=None: this is NOT a synthetic-channel env. The name is
    # resolvable + validatable now (D3.3); executing it before D3.4 wires the
    # BYO branch fails loud in _execute_group (the twin battery cannot run it).
    NoiseConfig(
        name="device_calibrated",
        channels=None,
        method="statevector",
        shots=4096,
        measurement_stats_interval=0,
        coupling_map_source="calibration",
        tier="device",
        description="Real Q50-calibrated noise (device_noise, statevector). "
                    "Executed by the BYO counts path (D3.4), not the twin battery.",
        source="device_calibrated",
    ),
]

# Lookup by name
NOISE_ENV_BY_NAME: dict[str, NoiseConfig] = {nc.name: nc for nc in NOISE_ENVIRONMENTS}

# Convenience
NOISELESS_ENVS = [nc for nc in NOISE_ENVIRONMENTS if nc.tier == "noiseless"]
NOISY_ENVS = [nc for nc in NOISE_ENVIRONMENTS if nc.tier != "noiseless"]
TIER_A_ENVS = [nc for nc in NOISE_ENVIRONMENTS if nc.tier == "A"]
TIER_B_ENVS = [nc for nc in NOISE_ENVIRONMENTS if nc.tier == "B"]


def get_active_channels_string(config: NoiseConfig) -> str:
    """Return a human-readable string of active channels."""
    if config.channels is None:
        return "none"
    active = [k for k, v in config.channels.items() if v]
    return ", ".join(active) if active else "none"


def get_env_names() -> list[str]:
    """Return ordered list of all environment names."""
    return [nc.name for nc in NOISE_ENVIRONMENTS]
