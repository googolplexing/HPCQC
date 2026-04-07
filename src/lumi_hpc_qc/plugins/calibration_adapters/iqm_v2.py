# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""IQM v2 calibration adapter.

Reads VTT Q50 and Aalto Q20 calibration JSON files (IQM format).
The JSON schema has top-level keys: calibration_set_id, timestamp,
device, qubits (per-qubit T1/T2/readout/gate error), two_qubit_gates
(per-pair CZ fidelity/error), and timing fields.

Phase E — RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0, System 1
"""

from __future__ import annotations

import json
from pathlib import Path

from lumi_hpc_qc.plugins.calibration_adapters.base import (
    AbstractCalibrationAdapter,
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)


class IQMv2Adapter(AbstractCalibrationAdapter):
    """Adapter for IQM JSON v2 calibration format.

    Handles VTT Q50 (53 qubits), Aalto Q20 (20 qubits), and
    future IQM devices that use the same schema.
    """

    name = "iqm_v2"

    def __init__(self, device_id: str = "", prefix: str = "") -> None:
        self._device_id = device_id
        self._prefix = prefix

    @property
    def device_prefix(self) -> str:
        return self._prefix

    def load(self, path: str) -> DeviceCalibration:
        """Load an IQM v2 calibration JSON file.

        Expected schema:
            {
                "calibration_set_id": str,
                "timestamp": str,
                "device": str,
                "qubits": {"QB1": {"t1_us": float, "t2_us": float,
                                    "readout_fidelity": float,
                                    "single_gate_error": float}, ...},
                "two_qubit_gates": {"QB1-QB2": {"cz_fidelity": float,
                                                 "cz_error": float}, ...},
                "single_gate_time_ns": float,
                "cz_gate_time_ns": float
            }
        """
        cal_path = Path(path)
        if not cal_path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")

        with open(cal_path) as f:
            raw = json.load(f)

        # Determine device identity
        device_name = raw.get("device", self._device_id or "unknown")
        device_id = self._device_id or device_name
        prefix = self._prefix or self._infer_prefix(device_name, raw)

        qubit_data = raw.get("qubits", {})
        gate_data = raw.get("two_qubit_gates", {})

        # Sort qubit names NUMERICALLY to match IQM backend convention.
        # Lexicographic sort gives QB1,QB10,QB11...QB19,QB2,QB20... — wrong.
        # IQM's dynamic quantum architecture assigns indices in numeric QB
        # order, skipping deactivated qubits (e.g. QB32 on Q50).
        # The sorted numeric order produces the correct index mapping:
        # QB1→0, QB2→1, ..., QB31→30, QB33→31, ..., QB54→52
        #
        # IMPORTANT (v1.1.1, RED-RESP-PACKING-v1.0 §3.3):
        # This mapping is calibration-set dependent. If VTT deactivates
        # different qubits in a future calibration cycle, the index
        # assignments change. Do NOT hardcode arithmetic offsets like
        # QB_N = index N-1 anywhere outside this adapter. Use the
        # index_to_qubit_name / qubit_name_to_index properties on
        # DeviceCalibration instead.
        #
        # Ref: VTT QX "Exploring the Device Quantum Architecture"
        # https://qx.vtt.fi/docs/advanced/dynamic-quantum-architecture.html
        qubit_names = sorted(
            qubit_data.keys(),
            key=lambda n: int(''.join(c for c in n if c.isdigit()) or '0'),
        )
        name_to_idx = {n: i for i, n in enumerate(qubit_names)}

        # Verify index continuity — indices must be 0..N-1 with no gaps
        assert len(qubit_names) == len(name_to_idx), (
            f"Duplicate qubit names in calibration: "
            f"{len(qubit_names)} names, {len(name_to_idx)} unique"
        )

        # Build normalized qubits
        qubits: dict[str, QubitCalibration] = {}
        for qname in qubit_names:
            qd = qubit_data[qname]
            qubits[qname] = QubitCalibration(
                name=qname,
                index=name_to_idx[qname],
                t1_us=qd.get("t1_us", 50.0),
                t2_us=qd.get("t2_us", 30.0),
                readout_fidelity=qd.get("readout_fidelity", 0.95),
                single_gate_error=qd.get("single_gate_error", 0.001),
            )

        # Build normalized gates + adjacency
        #
        # Connectivity source priority:
        # 1. Explicit qubit_connectivity from QX API architecture endpoint
        #    (authoritative — includes edges that may lack calibration data)
        # 2. Inferred from two_qubit_gates keys
        #    (backward compat with calibration files that lack the field)
        gates: dict[str, GateCalibration] = {}
        adjacency: dict[int, set[int]] = {i: set() for i in range(len(qubit_names))}

        # Step 1: Build gate calibrations from two_qubit_gates (fidelity data)
        for gate_key, gd in gate_data.items():
            parts = gate_key.split("-")
            if len(parts) != 2:
                continue
            q1_name, q2_name = parts
            if q1_name not in name_to_idx or q2_name not in name_to_idx:
                continue

            fidelity = gd.get("cz_fidelity", 1.0 - gd.get("cz_error", 0.005))
            error = gd.get("cz_error", 1.0 - fidelity)

            gates[gate_key] = GateCalibration(
                qubit_pair=(q1_name, q2_name),
                index_pair=(name_to_idx[q1_name], name_to_idx[q2_name]),
                fidelity=fidelity,
                error=error,
                gate_type="cz",
            )

        # Step 2: Build adjacency from explicit connectivity (preferred)
        # or fall back to gate keys (backward compat)
        explicit_connectivity = raw.get("qubit_connectivity")
        if explicit_connectivity:
            for pair in explicit_connectivity:
                if len(pair) != 2:
                    continue
                q1_name, q2_name = pair[0], pair[1]
                if q1_name not in name_to_idx or q2_name not in name_to_idx:
                    continue
                i, j = name_to_idx[q1_name], name_to_idx[q2_name]
                adjacency[i].add(j)
                adjacency[j].add(i)
                # NOTE: We do NOT create GateCalibration entries for edges
                # that lack calibration data. Adjacency is for topology
                # (does this edge physically exist?). Gates are for quality
                # (how good is this edge?). Assigning fake fidelity to
                # uncalibrated edges would bias the placement solver toward
                # unknown edges over measured-but-mediocre ones.
        else:
            # Backward compat: infer adjacency from calibrated gate keys
            for gate_key in gates:
                parts = gate_key.split("-")
                if len(parts) != 2:
                    continue
                q1_name, q2_name = parts
                if q1_name in name_to_idx and q2_name in name_to_idx:
                    i, j = name_to_idx[q1_name], name_to_idx[q2_name]
                    adjacency[i].add(j)
                    adjacency[j].add(i)

        # Determine topology name from structure
        topology = self._classify_topology(len(qubit_names), adjacency)

        return DeviceCalibration(
            device_id=device_id,
            device_prefix=prefix,
            num_qubits=len(qubit_names),
            topology_name=topology,
            qubits=qubits,
            gates=gates,
            adjacency=adjacency,
            single_gate_time_ns=raw.get("single_gate_time_ns", 25.0),
            two_qubit_gate_time_ns=raw.get("cz_gate_time_ns", 100.0),
            calibration_set_id=raw.get("calibration_set_id", ""),
            calibration_timestamp=raw.get("timestamp", ""),
            calibration_source_file=str(cal_path),
            is_synthetic=False,
        )

    def validate(self, calibration: DeviceCalibration) -> list[str]:
        """Validate IQM calibration data."""
        warnings = []

        for qname, qcal in calibration.qubits.items():
            if qcal.t1_us <= 0:
                warnings.append(f"{qname}: T1 ≤ 0 ({qcal.t1_us})")
            if qcal.t2_us <= 0:
                warnings.append(f"{qname}: T2 ≤ 0 ({qcal.t2_us})")
            if qcal.t2_us > 2 * qcal.t1_us:
                warnings.append(
                    f"{qname}: T2 ({qcal.t2_us}) > 2*T1 ({2*qcal.t1_us}) — "
                    f"violates physical constraint"
                )
            if not 0.5 <= qcal.readout_fidelity <= 1.0:
                warnings.append(
                    f"{qname}: readout fidelity out of range "
                    f"({qcal.readout_fidelity})"
                )

        for gkey, gcal in calibration.gates.items():
            if not 0.9 <= gcal.fidelity <= 1.0:
                warnings.append(
                    f"Gate {gkey}: fidelity out of typical range "
                    f"({gcal.fidelity})"
                )

        if calibration.num_qubits == 0:
            warnings.append("No qubits in calibration")

        return warnings

    def _infer_prefix(self, device_name: str, raw: dict) -> str:
        """Infer device prefix from device name and qubit count."""
        num_qubits = len(raw.get("qubits", {}))
        device_lower = device_name.lower()

        if "q50" in device_lower or num_qubits == 53:
            return "vtt_q50"
        elif "q20" in device_lower or num_qubits == 20:
            return "aalto_q20"
        elif "q100" in device_lower or num_qubits >= 90:
            return f"iqm_q{num_qubits}"
        else:
            return f"iqm_{num_qubits}q"

    def _classify_topology(
        self, num_qubits: int, adjacency: dict[int, set[int]]
    ) -> str:
        """Classify the device topology from its adjacency structure."""
        num_edges = sum(len(neighbors) for neighbors in adjacency.values()) // 2
        max_degree = max(
            (len(neighbors) for neighbors in adjacency.values()), default=0
        )

        if num_qubits == 53 and num_edges == 82:
            return "square_lattice"
        elif max_degree <= 2:
            return "linear"
        elif max_degree == 3:
            return "heavy_hex"
        else:
            return f"custom_{num_qubits}q_{num_edges}e"
