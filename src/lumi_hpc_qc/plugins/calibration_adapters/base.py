# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Abstract base class for calibration adapters.

Calibration adapters normalize vendor-specific calibration file formats
into a standard internal representation. Each QPU vendor (IQM, IBM,
Quantinuum, etc.) provides calibration data in a different JSON schema.
The adapter converts this into the standard fields that the placement
solver, noise model, and metadata pipeline consume.

To add a new device vendor:
  1. Create a new file in plugins/calibration_adapters/ (e.g., ibm_v1.py)
  2. Subclass AbstractCalibrationAdapter
  3. Implement all abstract methods
  4. Set the `name` class attribute
  5. The plugin registry auto-discovers it

Phase E — RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0, System 1
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class QubitCalibration:
    """Normalized per-qubit calibration data."""

    name: str
    index: int
    t1_us: float
    t2_us: float
    readout_fidelity: float
    single_gate_error: float = 0.001


@dataclass(frozen=True)
class GateCalibration:
    """Normalized per-gate calibration data."""

    qubit_pair: tuple[str, str]
    index_pair: tuple[int, int]
    fidelity: float
    error: float
    gate_type: str = "cz"


@dataclass
class DeviceCalibration:
    """Complete normalized calibration for one device.

    This is the standard internal representation that all framework
    components consume. The calibration adapter's job is to produce
    this from vendor-specific JSON.
    """

    device_id: str
    device_prefix: str
    num_qubits: int
    topology_name: str

    qubits: dict[str, QubitCalibration] = field(default_factory=dict)
    gates: dict[str, GateCalibration] = field(default_factory=dict)

    # Coupling map: adjacency as {qubit_index: [neighbor_indices]}
    adjacency: dict[int, set[int]] = field(default_factory=dict)

    # Timing
    single_gate_time_ns: float = 25.0
    two_qubit_gate_time_ns: float = 100.0

    # Provenance
    calibration_set_id: str = ""
    calibration_timestamp: str = ""
    calibration_source_file: str = ""
    is_synthetic: bool = False
    synthetic_perturbation: str = ""

    @property
    def qubit_names(self) -> list[str]:
        """Sorted list of qubit names."""
        return sorted(self.qubits.keys())

    @property
    def qubit_name_to_index(self) -> dict[str, int]:
        """Map qubit name to integer index."""
        return {q.name: q.index for q in self.qubits.values()}

    @property
    def index_to_qubit_name(self) -> dict[int, str]:
        """Map integer index to qubit name."""
        return {q.index: q.name for q in self.qubits.values()}

    @property
    def coupling_edges(self) -> list[tuple[str, str]]:
        """List of (q1, q2) coupling pairs, deduplicated, sorted."""
        edges = set()
        for gate_cal in self.gates.values():
            q1, q2 = gate_cal.qubit_pair
            edge = (min(q1, q2), max(q1, q2))
            edges.add(edge)
        return sorted(edges)

    def gate_fidelity(self, idx_a: int, idx_b: int) -> float:
        """Get gate fidelity between two qubit indices."""
        name_a = self.index_to_qubit_name.get(idx_a, "")
        name_b = self.index_to_qubit_name.get(idx_b, "")
        key_1 = f"{name_a}-{name_b}"
        key_2 = f"{name_b}-{name_a}"
        if key_1 in self.gates:
            return self.gates[key_1].fidelity
        if key_2 in self.gates:
            return self.gates[key_2].fidelity
        return 0.0


class AbstractCalibrationAdapter(ABC):
    """Abstract base for calibration file adapters.

    Each adapter knows how to read one vendor's calibration format
    and produce a DeviceCalibration. The adapter is a plugin —
    auto-discovered by name.
    """

    name: str = ""  # Subclass must set: "iqm_v2", "ibm_v1", etc.

    @abstractmethod
    def load(self, path: str) -> DeviceCalibration:
        """Load a calibration file and return normalized data.

        Args:
            path: Path to the vendor-specific calibration JSON file.

        Returns:
            DeviceCalibration with all fields populated.

        Raises:
            FileNotFoundError: If the calibration file doesn't exist.
            ValueError: If the file format is invalid or missing fields.
        """
        ...

    @abstractmethod
    def validate(self, calibration: DeviceCalibration) -> list[str]:
        """Validate a DeviceCalibration for completeness.

        Returns:
            List of warning messages. Empty list = valid.
        """
        ...

    @property
    @abstractmethod
    def device_prefix(self) -> str:
        """Device prefix for HDF5 group naming.

        e.g., 'vtt_q50', 'aalto_q20', 'it4i_vlq'
        """
        ...
