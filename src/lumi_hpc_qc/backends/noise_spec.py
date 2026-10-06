# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Noise-channel selection for the device-calibrated backend.

This module is intentionally free of any qiskit/qiskit-aer import: it is the
small, reusable vocabulary that lets ANY HPCQC experiment (not just the Floquet
driver) say which noise channels the device-calibrated simulation should apply.
The device-calibrated path always decomposes to native gates, routes, and
schedules; this spec only toggles which physical noise sources are switched on,
so a researcher can run the full model or ablate one channel at a time on the
SAME native circuit (an honest error budget).

Channels (canonical name -> what it models):
  single_qubit_depolarizing : control error on each native 1q gate (r/sx/x, rz)
  two_qubit_depolarizing    : control error on each native CZ
  readout                   : measurement bit-flip error
  thermal_relaxation        : T1/T2 decoherence -- the loss of quantum
                              information simply because real time passes. Two
                              physical effects: T1 (energy relaxation) is how
                              fast an excited qubit decays toward |0>, and T2
                              (dephasing) is how fast a superposition loses its
                              phase. This channel applies that decay in the two
                              places a qubit experiences time:
                                * gate-duration relaxation -- every native gate
                                  occupies the qubit for a fixed time (~20 ns for
                                  a 1q gate, ~60 ns for a CZ), so the qubit
                                  decoheres for that long while the gate runs.
                                  The duration is fixed, so this part is baked
                                  into the static noise model ("resident").
                                * idle/delay relaxation -- while a qubit waits
                                  for its neighbours (the Delays the ALAP
                                  scheduler inserts), it decoheres for the REAL
                                  idle duration. That varies per circuit, so a
                                  duration-aware pass applies it at run time.
                              T1 and T2 are PER-QUBIT calibration values (see the
                              overrides below for how to replace them). This is
                              the only channel that can become a genuine
                              non-unitary Kraus map: when a qubit's T2 > T1 the
                              relaxation cannot be written as a probabilistic mix
                              of resets/Paulis and must be a true Kraus channel.
                              That is why device-calibrated pins
                              method="statevector" -- doing so makes Aer
                              precompute the channel's canonical Kraus
                              deterministically, so per-shot statevector sampling
                              of a T2 > T1 qubit never hits an empty Kraus. (With
                              the simulator left on "automatic" that precompute
                              can be skipped, which is the "Kraus is empty" crash
                              this design avoids.)

Grammar: the value of --noise is "all", "none", or a comma-separated list of
channel tokens. Concrete examples (left = what you pass to --noise; right =
what the device-calibrated run applies):

  all                                     every channel on (the default)
  none                                    native circuit, zero noise
  1q                                      single-qubit depolarizing only
  2q                                      two-qubit CZ depolarizing only
  measurement                             readout error only
  thermal_relaxation_error                T1/T2 relaxation only (per-qubit, from JSON)
  1q,2q                                   both depolarizing channels; no readout/relaxation
  measurement,1q                          readout + single-qubit depolarizing
  1q,2q,measurement                       full model EXCEPT relaxation
  thermal_relaxation_error(t2_us=12)      relaxation; every qubit forced to T2 = 12 us
  1q,thermal_relaxation_error(t2_us=12)   1q depolarizing + relaxation (uniform T2 = 12 us)
  thermal_relaxation_error(t1_us=6.4,t2_us=12.0,dt_ns=60)
                                          relaxation; every qubit T1 = 6.4 us,
                                          T2 = 12 us, gate-relaxation duration 60 ns

Tokens are case-insensitive and whitespace-tolerant. Aliases:
  1q  = sq = single_qubit_depolarizing
  2q  = cz = two_qubit_depolarizing
  measurement = measure = meas = ro = readout
  thermal = relaxation = t1t2 = thermal_relaxation_error
An unknown token raises ValueError listing the valid tokens, so a typo fails
loudly at submit time rather than silently dropping a channel.

Thermal overrides -- valid ONLY on the thermal_relaxation_error token, written
as keyword args, e.g. thermal_relaxation_error(t1_us=6.4,t2_us=12,dt_ns=60).
All three are OPTIONAL and UNIFORM (one value applied to every selected qubit);
any you omit keeps that qubit's per-qubit calibrated value.
  t1_us   T1 for every qubit, in microseconds
  t2_us   T2 for every qubit, in microseconds (still capped at the 2*T1 ceiling)
  dt_ns   gate-relaxation DURATION, in nanoseconds -- replaces the ~20 ns / 60 ns
          native gate times used by the RESIDENT (gate-time) relaxation only; it
          does NOT change the circuit schedule or the idle/delay relaxation.
Use overrides for sensitivity/ablation sweeps ("what if every qubit had
T2 = 12 us"), not for a faithful per-qubit model.

Default T1/T2 source (when you do NOT override): each selected qubit uses its
OWN calibrated T1 (the "t1_us" field) and its own T2. WHICH T2 is read is set by
the SEPARATE --t2-mode flag, not by --noise:
  --t2-mode ramsey  (default)  reads "t2_us"      (free-induction T2*)
  --t2-mode echo               reads "t2_echo_us" (Hahn-echo T2)
T2 is always capped at the physical maximum 2*T1.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# Canonical channel names.
SINGLE_QUBIT_DEPOLARIZING = "single_qubit_depolarizing"
TWO_QUBIT_DEPOLARIZING = "two_qubit_depolarizing"
READOUT = "readout"
THERMAL_RELAXATION = "thermal_relaxation"

CANONICAL_CHANNELS = (
    SINGLE_QUBIT_DEPOLARIZING,
    TWO_QUBIT_DEPOLARIZING,
    READOUT,
    THERMAL_RELAXATION,
)

# Friendly aliases -> canonical name. Keys are matched lower-cased.
_ALIASES = {
    "1q": SINGLE_QUBIT_DEPOLARIZING,
    "sq": SINGLE_QUBIT_DEPOLARIZING,
    "single_qubit_depolarizing": SINGLE_QUBIT_DEPOLARIZING,
    "single-qubit-depolarizing": SINGLE_QUBIT_DEPOLARIZING,
    "depol1": SINGLE_QUBIT_DEPOLARIZING,
    "2q": TWO_QUBIT_DEPOLARIZING,
    "tq": TWO_QUBIT_DEPOLARIZING,
    "two_qubit_depolarizing": TWO_QUBIT_DEPOLARIZING,
    "two-qubit-depolarizing": TWO_QUBIT_DEPOLARIZING,
    "depol2": TWO_QUBIT_DEPOLARIZING,
    "cz": TWO_QUBIT_DEPOLARIZING,
    "readout": READOUT,
    "measurement": READOUT,
    "measure": READOUT,
    "meas": READOUT,
    "ro": READOUT,
    "thermal_relaxation": THERMAL_RELAXATION,
    "thermal_relaxation_error": THERMAL_RELAXATION,
    "thermal": THERMAL_RELAXATION,
    "relaxation": THERMAL_RELAXATION,
    "t1t2": THERMAL_RELAXATION,
    "t1_t2": THERMAL_RELAXATION,
}

# token like  name(...)  -> capture name and the inner argument string
_CALL_RE = re.compile(r"^([A-Za-z0-9_\-]+)\s*\((.*)\)$")


def _split_top_level(value: str) -> list[str]:
    """Split on commas that are NOT inside parentheses, so a token like
    'thermal(t1_us=6.4,t2_us=12)' stays intact while '1q,2q' splits in two."""
    tokens: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in value:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            tokens.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        tokens.append("".join(buf))
    return tokens


@dataclass(frozen=True)
class NoiseSpec:
    """Which device-calibrated noise channels are active, plus optional
    thermal overrides. Defaults to the full model (every channel on)."""

    single_qubit_depolarizing: bool = True
    two_qubit_depolarizing: bool = True
    readout: bool = True
    thermal_relaxation: bool = True
    # Optional explicit overrides for the thermal channel. None -> use the
    # values derived from the calibration JSON / VTT durations.
    thermal_overrides: dict = field(default_factory=dict)

    @property
    def any_genuine_kraus_possible(self) -> bool:
        """True if a configuration could yield a non-unitary Kraus channel
        (i.e. thermal is on). The simulator pins statevector in this case."""
        return self.thermal_relaxation

    def active_channels(self) -> tuple[str, ...]:
        out = []
        if self.single_qubit_depolarizing:
            out.append(SINGLE_QUBIT_DEPOLARIZING)
        if self.two_qubit_depolarizing:
            out.append(TWO_QUBIT_DEPOLARIZING)
        if self.readout:
            out.append(READOUT)
        if self.thermal_relaxation:
            out.append(THERMAL_RELAXATION)
        return tuple(out)

    def describe(self) -> str:
        chans = self.active_channels()
        body = ",".join(chans) if chans else "none"
        if self.thermal_overrides:
            body += f" (thermal overrides: {self.thermal_overrides})"
        return body


def _parse_thermal_args(arg_str: str) -> dict:
    """Parse the inside of thermal_relaxation_error(...) into overrides.

    Accepts keyword form 't1_us=6.4,t2_us=12,dt_ns=60' (recommended) or up to
    three positional floats interpreted as (t1_us, t2_us, dt_ns). Empty -> {}.
    """
    arg_str = arg_str.strip()
    if not arg_str:
        return {}
    overrides: dict = {}
    positional: list[float] = []
    for part in arg_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            key, _, val = part.partition("=")
            key = key.strip().lower()
            if key not in ("t1_us", "t2_us", "dt_ns"):
                raise ValueError(
                    f"unknown thermal override '{key}' "
                    f"(valid: t1_us, t2_us, dt_ns)"
                )
            overrides[key] = float(val.strip())
        else:
            positional.append(float(part))
    if positional:
        for key, val in zip(("t1_us", "t2_us", "dt_ns"), positional):
            overrides.setdefault(key, val)
    return overrides


def parse_noise_spec(value: str) -> NoiseSpec:
    """Parse a --noise value string into a NoiseSpec.

    >>> parse_noise_spec("all").active_channels()
    ('single_qubit_depolarizing', 'two_qubit_depolarizing', 'readout', 'thermal_relaxation')
    >>> parse_noise_spec("1q,2q").active_channels()
    ('single_qubit_depolarizing', 'two_qubit_depolarizing')
    >>> parse_noise_spec("none").active_channels()
    ()
    """
    if value is None:
        raise ValueError("noise spec is None; pass 'all' for the full model")
    raw = value.strip().lower()
    if raw in ("all", "*"):
        return NoiseSpec(True, True, True, True)
    if raw in ("none", "noiseless", ""):
        return NoiseSpec(False, False, False, False)

    enabled = {c: False for c in CANONICAL_CHANNELS}
    thermal_overrides: dict = {}

    for token in _split_top_level(value):
        token = token.strip()
        if not token:
            continue
        name = token.lower()
        call = _CALL_RE.match(token)
        if call:
            name = call.group(1).lower()
            inner = call.group(2)
        else:
            inner = None
        if name not in _ALIASES:
            valid = sorted(set(_ALIASES) | {"all", "none"})
            raise ValueError(
                f"unknown noise channel '{token}'. Valid tokens: {valid}"
            )
        canonical = _ALIASES[name]
        enabled[canonical] = True
        if inner is not None:
            if canonical != THERMAL_RELAXATION:
                raise ValueError(
                    f"channel '{token}' does not take arguments; only "
                    f"thermal_relaxation_error(...) does"
                )
            thermal_overrides.update(_parse_thermal_args(inner))

    return NoiseSpec(
        single_qubit_depolarizing=enabled[SINGLE_QUBIT_DEPOLARIZING],
        two_qubit_depolarizing=enabled[TWO_QUBIT_DEPOLARIZING],
        readout=enabled[READOUT],
        thermal_relaxation=enabled[THERMAL_RELAXATION],
        thermal_overrides=thermal_overrides,
    )
