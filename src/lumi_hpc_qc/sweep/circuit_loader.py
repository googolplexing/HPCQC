# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Circuit loader — BYO circuit ingestion for the sweep engine.

Loads quantum circuits from QPY files, QASM files/strings, or Python
scripts. Extracts qubit connectivity for the placement solver.
Detects whether a circuit is parameterized (VQE) or fixed (eval-only).

RED-SPEC-002 §7 — BYO Circuit Ingestion
RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0 §4.3

Supported formats (in priority order):
  1. QPY binary (Qiskit native, preserves all metadata)
  2. Python script (function returning QuantumCircuit)
  3. QASM 3.0 file
  4. QASM 2.0 file (legacy fallback)
  5. Inline QASM string
"""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass, field
from typing import Any

from qiskit import QuantumCircuit


@dataclass
class LoadedCircuit:
    """A loaded BYO circuit with extracted metadata.

    Attributes:
        circuit: The Qiskit QuantumCircuit.
        num_qubits: Number of qubits.
        num_parameters: Number of unbound Parameters (0 = eval-only).
        is_parameterized: True if circuit has unbound Parameters.
        connectivity: List of (i, j) qubit pairs connected by 2q gates.
        source: Description of where the circuit was loaded from.
        gate_counts: Dict of gate name → count.
        depth: Circuit depth before transpilation.
    """
    circuit: QuantumCircuit
    num_qubits: int = 0
    num_parameters: int = 0
    is_parameterized: bool = False
    connectivity: list[tuple[int, int]] = field(default_factory=list)
    source: str = ""
    gate_counts: dict[str, int] = field(default_factory=dict)
    depth: int = 0


def load_circuit(
    *,
    qpy_file: str | None = None,
    qasm_file: str | None = None,
    qasm_string: str | None = None,
    script_file: str | None = None,
    script_function: str = "build_circuit",
) -> LoadedCircuit:
    """Load a quantum circuit from various sources.

    Exactly one source must be provided. The circuit is analyzed for
    qubit connectivity (2q gate pairs) and parameterization state.

    Args:
        qpy_file: Path to a QPY binary file.
        qasm_file: Path to a QASM file (2.0 or 3.0).
        qasm_string: Inline QASM string.
        script_file: Path to a Python script containing a circuit builder.
        script_function: Name of the function in the script (default: "build_circuit").

    Returns:
        LoadedCircuit with extracted metadata.

    Raises:
        ValueError: If zero or multiple sources provided, or loading fails.
    """
    sources = [
        ("qpy_file", qpy_file),
        ("qasm_file", qasm_file),
        ("qasm_string", qasm_string),
        ("script_file", script_file),
    ]
    provided = [(name, val) for name, val in sources if val is not None]

    if len(provided) == 0:
        raise ValueError(
            "No circuit source provided. Supply one of: "
            "qpy_file, qasm_file, qasm_string, script_file"
        )
    if len(provided) > 1:
        raise ValueError(
            f"Multiple circuit sources provided: {[n for n, _ in provided]}. "
            "Supply exactly one."
        )

    source_name, source_val = provided[0]

    if source_name == "qpy_file":
        qc, desc = _load_qpy(source_val)
    elif source_name == "qasm_file":
        qc, desc = _load_qasm_file(source_val)
    elif source_name == "qasm_string":
        qc, desc = _load_qasm_string(source_val)
    elif source_name == "script_file":
        qc, desc = _load_script(source_val, script_function)
    else:
        raise ValueError(f"Unknown source: {source_name}")

    return _analyze(qc, desc)


def load_from_config(config: dict[str, Any]) -> LoadedCircuit:
    """Load a circuit from a sweep YAML config dict.

    Expected keys (one of):
        circuit_file: path to QPY or QASM file (auto-detected by extension)
        circuit_script: path to Python script
        circuit_function: function name in script (default: build_circuit)
        circuit_qasm: inline QASM string

    RED-SPEC-002 §7.3 — YAML Config Interface
    """
    circuit_file = config.get("circuit_file")
    circuit_script = config.get("circuit_script")
    circuit_qasm = config.get("circuit_qasm")
    circuit_function = config.get("circuit_function", "build_circuit")

    if circuit_file is not None:
        ext = os.path.splitext(circuit_file)[1].lower()
        if ext == ".qpy":
            return load_circuit(qpy_file=circuit_file)
        else:
            return load_circuit(qasm_file=circuit_file)
    elif circuit_script is not None:
        return load_circuit(
            script_file=circuit_script,
            script_function=circuit_function,
        )
    elif circuit_qasm is not None:
        return load_circuit(qasm_string=circuit_qasm)
    else:
        raise ValueError(
            "No circuit source in config. Provide one of: "
            "circuit_file, circuit_script, circuit_qasm"
        )


# ═══════════════════════════════════════════════════════════════════════
# Private loaders
# ═══════════════════════════════════════════════════════════════════════

def _load_qpy(path: str) -> tuple[QuantumCircuit, str]:
    """Load a circuit from QPY binary file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"QPY file not found: {path}")
    from qiskit.qpy import load as qpy_load
    with open(path, "rb") as f:
        circuits = qpy_load(f)
    if not circuits:
        raise ValueError(f"QPY file contains no circuits: {path}")
    return circuits[0], f"qpy:{os.path.basename(path)}"


def _load_qasm_file(path: str) -> tuple[QuantumCircuit, str]:
    """Load a circuit from QASM file (tries 3.0, falls back to 2.0)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"QASM file not found: {path}")
    try:
        from qiskit.qasm3 import load as qasm3_load
        qc = qasm3_load(path)
        return qc, f"qasm3:{os.path.basename(path)}"
    except Exception:
        qc = QuantumCircuit.from_qasm_file(path)
        return qc, f"qasm2:{os.path.basename(path)}"


def _load_qasm_string(qasm_str: str) -> tuple[QuantumCircuit, str]:
    """Load a circuit from an inline QASM string."""
    try:
        from qiskit.qasm3 import loads as qasm3_loads
        qc = qasm3_loads(qasm_str)
        return qc, "qasm3:inline"
    except Exception:
        qc = QuantumCircuit.from_qasm_str(qasm_str)
        return qc, "qasm2:inline"


def _load_script(path: str, function_name: str) -> tuple[QuantumCircuit, str]:
    """Load a circuit by calling a function in a Python script.

    The script must define a function that returns a QuantumCircuit.
    The function is called with no arguments.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Script file not found: {path}")

    spec = importlib.util.spec_from_file_location("byo_circuit_script", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load Python module from: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, function_name):
        raise ValueError(
            f"Script {path} has no function '{function_name}'. "
            f"Available: {[n for n in dir(module) if not n.startswith('_')]}"
        )

    fn = getattr(module, function_name)
    qc = fn()
    if not isinstance(qc, QuantumCircuit):
        raise TypeError(
            f"{function_name}() returned {type(qc).__name__}, "
            f"expected QuantumCircuit"
        )
    return qc, f"script:{os.path.basename(path)}:{function_name}"


# ═══════════════════════════════════════════════════════════════════════
# Analysis
# ═══════════════════════════════════════════════════════════════════════

def _analyze(qc: QuantumCircuit, source: str) -> LoadedCircuit:
    """Analyze a loaded circuit: extract connectivity, detect parameters."""
    connectivity = extract_connectivity(qc)
    gate_counts = dict(qc.count_ops())
    n_params = qc.num_parameters

    return LoadedCircuit(
        circuit=qc,
        num_qubits=qc.num_qubits,
        num_parameters=n_params,
        is_parameterized=(n_params > 0),
        connectivity=connectivity,
        source=source,
        gate_counts=gate_counts,
        depth=qc.depth(),
    )


def extract_connectivity(qc: QuantumCircuit) -> list[tuple[int, int]]:
    """Extract the set of qubit pairs connected by 2-qubit gates.

    Returns sorted unique pairs (i, j) where i < j, representing
    the circuit's connectivity graph for the placement solver.
    """
    pairs = set()
    for instruction in qc.data:
        if instruction.operation.num_qubits == 2:
            q0 = qc.find_bit(instruction.qubits[0]).index
            q1 = qc.find_bit(instruction.qubits[1]).index
            if q0 > q1:
                q0, q1 = q1, q0
            pairs.add((q0, q1))
    return sorted(pairs)
