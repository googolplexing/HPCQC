# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Example BYO circuit factory — Floquet DTC (SPEC-002 §7.5.6).

A pure, keyword-only circuit factory: it receives all parameters as data and
draws no randomness. The engine partitions these across the YAML blocks:
  grid:     num_kicks        (swept)
  fixed:    epsilon, num_qubits
  disorder: hz_angles, Jzz_angles, init_bit_array   (per seed, from JSON)

Gate body is an inline of floquet_runner.apply_one_floquet_period
(rx(h_x) -> rz(hz) -> rzz(Jzz)) and build_circuit (X-init on init_bit_array==1,
num_kicks periods, measure-all). epsilon is first-class so it can be promoted to
a grid axis with a YAML-only edit; h_x = (1-epsilon)*pi is computed internally.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit


def build_circuit(
    *,
    num_kicks: int,
    epsilon: float,
    num_qubits: int,
    hz_angles,
    Jzz_angles,
    init_bit_array,
) -> QuantumCircuit:
    h_x = (1.0 - epsilon) * np.pi
    qc = QuantumCircuit(num_qubits, num_qubits)
    for w in range(num_qubits):
        if init_bit_array[w] == 1:
            qc.x(w)
    for _ in range(num_kicks):
        for w in range(num_qubits):
            qc.rx(h_x, w)
        for w in range(num_qubits):
            qc.rz(hz_angles[w], w)
        for w in range(num_qubits - 1):
            qc.rzz(Jzz_angles[w], w, w + 1)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc
