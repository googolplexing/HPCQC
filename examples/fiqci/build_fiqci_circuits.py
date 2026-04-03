#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""FiQCI example circuits for BYO validation.

Builds non-parameterized QuantumCircuits inspired by FiQCI/fiqci-examples:
GHZ state preparation, Bell pairs, and multi-qubit entanglement. These
are characterization circuits — no optimizer, no parameters, one execution
per placement per noise config.

Usage as script (generates QPY files):
    python examples/fiqci/build_fiqci_circuits.py

Usage as circuit builder (for BYO circuit_loader):
    from examples.fiqci.build_fiqci_circuits import build_circuit
    qc = build_circuit()  # returns GHZ-3q by default

RED-SPEC-002 §16 VE14 — FiQCI GHZ 3q circuit validation
"""

from __future__ import annotations

import os
from pathlib import Path

from qiskit import QuantumCircuit


def build_ghz(num_qubits: int = 3) -> QuantumCircuit:
    """Build a GHZ state preparation circuit.

    |GHZ_n⟩ = (|00...0⟩ + |11...1⟩) / √2

    For n qubits: H on q[0], then CNOT chain q[0]→q[1]→...→q[n-1].
    """
    qc = QuantumCircuit(num_qubits, name=f"ghz_{num_qubits}q")
    qc.h(0)
    for i in range(num_qubits - 1):
        qc.cx(i, i + 1)
    return qc


def build_bell() -> QuantumCircuit:
    """Build a Bell pair circuit.

    |Φ+⟩ = (|00⟩ + |11⟩) / √2
    """
    qc = QuantumCircuit(2, name="bell_2q")
    qc.h(0)
    qc.cx(0, 1)
    return qc


def build_star_entanglement(num_qubits: int = 4) -> QuantumCircuit:
    """Build a star entanglement circuit — hub connected to all leaves.

    H on hub (q[0]), CNOT from hub to each leaf.
    Connectivity: star graph K_{1,n-1}.
    """
    qc = QuantumCircuit(num_qubits, name=f"star_{num_qubits}q")
    qc.h(0)
    for i in range(1, num_qubits):
        qc.cx(0, i)
    return qc


def build_circuit() -> QuantumCircuit:
    """Default circuit builder for BYO circuit_loader.

    Returns GHZ-3q for compatibility with:
        circuit_loader.load_circuit(script_file="build_fiqci_circuits.py")
    """
    return build_ghz(3)


# ═══════════════════════════════════════════════════════════════════════
# QPY file generation
# ═══════════════════════════════════════════════════════════════════════

CIRCUITS = {
    "bell_2q": build_bell,
    "ghz_3q": lambda: build_ghz(3),
    "ghz_4q": lambda: build_ghz(4),
    "ghz_5q": lambda: build_ghz(5),
    "star_4q": lambda: build_star_entanglement(4),
}


def generate_qpy_files(output_dir: str | None = None) -> dict[str, str]:
    """Generate QPY files for all FiQCI example circuits.

    Args:
        output_dir: Directory for QPY files. Default: same directory as this script.

    Returns:
        Dict of {circuit_name: qpy_file_path}.
    """
    from qiskit.qpy import dump

    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    paths = {}

    for name, builder in CIRCUITS.items():
        qc = builder()
        qpy_path = os.path.join(output_dir, f"{name}.qpy")
        with open(qpy_path, "wb") as f:
            dump([qc], f)
        paths[name] = qpy_path

    return paths


if __name__ == "__main__":
    paths = generate_qpy_files()
    for name, path in paths.items():
        print(f"  {name}: {path}")
    print(f"\nGenerated {len(paths)} QPY files")
