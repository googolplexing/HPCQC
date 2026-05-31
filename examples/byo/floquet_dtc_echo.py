# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""BYO circuit factory — Floquet DTC with echo (SPEC-002 §7.5, DEBT D7).

Derived from the researcher reference `Floquet_DTC_AK10_echo.py`. Like the
single-observable `floquet_dtc.py`, this is a pure, keyword-only factory that
draws no randomness — the engine supplies all parameters as data:

  grid:     num_kicks        (swept)
  fixed:    epsilon, num_qubits
  disorder: hz_angles, Jzz_angles, init_bit_array   (per seed, from JSON)

The difference: AK10 defines TWO observables per (seed, num_kicks) point,
  * autocorrelator A(0)A(T): forward num_kicks Floquet periods, then measure;
  * echo A_0: forward num_kicks periods then num_kicks CONJUGATE (time-
    reversed, negated-angle) periods, then measure; the echo VALUE is
    sqrt(|autocorrelator-of-the-doubled-circuit|).
and the headline signal is the per-kick ratio A(0)A(T) / A_0, which divides out
the decoherence envelope (the echo measures signal lost to noise alone under
perfect dynamical reversal) to isolate the genuine DTC response.

GATE BASIS NOTE (important for native-basis transpilation):
This factory emits LOGICAL rx / rz / rzz gates, identical to AK10 and to the
single-observable factory. It does NOT pre-transpile to the device native basis
(prx / cz). Native-basis lowering happens downstream and per-arm:
  * device_calibrated arm: prepare.py transpiles logical -> native (prx, cz)
    with calibrated durations + ALAP scheduling + PadDelay idle insertion
    (the path fixed at 90d329d), then attaches the noise model;
  * noiseless arm: the logical circuit runs on a statevector simulator with no
    hardware-basis transpilation and no noise model (the clean reference).
Keeping the factory in logical gates is therefore correct for BOTH arms; the
arms diverge at the engine/backend level, not here.

TWO ENTRY POINTS (see the design doc for how the engine consumes them):
  * build_circuit       — the autocorrelator circuit (signature-compatible with
                          the single-observable factory; lets this script be
                          used as a drop-in for the existing single-observable
                          path if echo support is not wired yet);
  * build_circuit_echo  — the doubled (forward + conjugate) echo circuit.
A future multi-observable engine calls BOTH per task and derives the ratio in
the analysis stage (the ratio is NOT computed here because it requires the
measured counts from both circuits, which only exist post-simulation).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def _apply_forward_period(qc, h_x, num_qubits, hz_angles, Jzz_angles):
    """One forward Floquet period: rx(h_x) -> rz(hz) -> rzz(Jzz/2).

    Inlines AK10.apply_one_floquet_period exactly (same gate order, same
    angle signs). The Jzz coupling is applied as Jzz/2 per period, matching
    the researcher's corrected Floquet_DTC_AK10_echo.py (the /2 is a build-time
    rescaling; the disorder JSON's drawn Jzz_angles are unchanged).
    """
    for w in range(num_qubits):
        qc.rx(h_x, w)
    for w in range(num_qubits):
        qc.rz(hz_angles[w], w)
    for w in range(num_qubits - 1):
        qc.rzz(Jzz_angles[w] / 2, w, w + 1)


def _apply_conjugate_period(qc, h_x, num_qubits, hz_angles, Jzz_angles):
    """One time-reversed Floquet period: rzz(-Jzz/2) -> rz(-hz) -> rx(-h_x).

    Inlines AK10.apply_one_Floquet_period_conjugate exactly: gates in reverse
    order with negated angles. This is the dagger of _apply_forward_period.
    """
    for w in range(num_qubits - 1):
        qc.rzz(-Jzz_angles[w] / 2, w, w + 1)
    for w in range(num_qubits):
        qc.rz(-hz_angles[w], w)
    for w in range(num_qubits):
        qc.rx(-h_x, w)


def build_circuit(
    *,
    num_kicks: int,
    epsilon: float,
    num_qubits: int,
    hz_angles,
    Jzz_angles,
    init_bit_array,
) -> QuantumCircuit:
    """Autocorrelator circuit: X-init, num_kicks forward periods, measure-all.

    Signature-compatible with floquet_dtc.build_circuit, but NOT identical in
    physics: this echo factory applies the coupling as Jzz/2 per period
    (AK10-derived; see _apply_forward_period), whereas floquet_dtc.build_circuit
    applies Jzz with no /2 (AK7-derived). They diverge by a factor of two in the
    rzz drive and are therefore NOT interchangeable -- in particular the W1.6
    gate MUST use floquet_dtc.build_circuit, never this one. This builder exists
    only so the echo script is a self-contained superset for the AK10
    autocorrelator-vs-echo ratio; it is not a drop-in for the gate factory.
    """
    h_x = (1.0 - epsilon) * np.pi
    qc = QuantumCircuit(num_qubits, num_qubits)
    for w in range(num_qubits):
        if init_bit_array[w] == 1:
            qc.x(w)
    for _ in range(num_kicks):
        _apply_forward_period(qc, h_x, num_qubits, hz_angles, Jzz_angles)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def build_circuit_echo(
    *,
    num_kicks: int,
    epsilon: float,
    num_qubits: int,
    hz_angles,
    Jzz_angles,
    init_bit_array,
) -> QuantumCircuit:
    """Echo circuit: X-init, num_kicks forward + num_kicks conjugate, measure.

    Inlines AK10.build_echo_circuit. Under noiseless, perfect dynamical
    reversal returns the system to the initial state, so the autocorrelator of
    THIS circuit is ~1 at every kick; under noise it decays, and that decay is
    the decoherence envelope the ratio divides out. The sqrt(|.|) that turns the
    measured autocorrelator into the echo VALUE A_0 is applied in the analysis
    stage (post-simulation), not here — this function only builds the circuit.
    """
    h_x = (1.0 - epsilon) * np.pi
    qc = QuantumCircuit(num_qubits, num_qubits)
    for w in range(num_qubits):
        if init_bit_array[w] == 1:
            qc.x(w)
    for _ in range(num_kicks):
        _apply_forward_period(qc, h_x, num_qubits, hz_angles, Jzz_angles)
    for _ in range(num_kicks):
        _apply_conjugate_period(qc, h_x, num_qubits, hz_angles, Jzz_angles)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc
