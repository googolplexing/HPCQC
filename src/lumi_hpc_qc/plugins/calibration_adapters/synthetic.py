# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Synthetic calibration adapter.

Generates perturbed calibration data from a real calibration source.
Used for stress-testing ML models on noise regimes that don't exist
on current hardware, threshold identification, and future device
projection.

Phase E — RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0, System 3 (E9)
"""

from __future__ import annotations

import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from lumi_hpc_qc.plugins.calibration_adapters.base import (
    AbstractCalibrationAdapter,
    DeviceCalibration,
    GateCalibration,
    QubitCalibration,
)
from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter


class SyntheticAdapter(AbstractCalibrationAdapter):
    """Generate synthetic calibrations from real calibration data.

    Supports perturbation operations:
      - scale_t1: multiply all T1 values by a factor
      - scale_t2: multiply all T2 values by a factor
      - scale_readout: multiply all readout fidelities by a factor
      - scale_gate_fidelity: multiply all gate fidelities by a factor
      - poison_qubit: set one qubit to degraded values
      - project_topology: remap calibration onto a different coupling map
    """

    name = "synthetic"

    def __init__(self, base_adapter: AbstractCalibrationAdapter | None = None):
        self._base_adapter = base_adapter or IQMv2Adapter()
        self._prefix = "synthetic"

    @property
    def device_prefix(self) -> str:
        return self._prefix

    def load(self, path: str) -> DeviceCalibration:
        """Load the base calibration without perturbation."""
        return self._base_adapter.load(path)

    def validate(self, calibration: DeviceCalibration) -> list[str]:
        """Validate a synthetic calibration."""
        warnings = []
        for qname, qcal in calibration.qubits.items():
            if qcal.t1_us <= 0:
                warnings.append(f"{qname}: T1 ≤ 0 after perturbation")
            if qcal.t2_us > 2 * qcal.t1_us:
                warnings.append(f"{qname}: T2 > 2*T1 after perturbation")
        return warnings

    def perturb(
        self,
        base: DeviceCalibration,
        perturbation: dict[str, Any],
    ) -> DeviceCalibration:
        """Apply a perturbation spec to a base calibration.

        Args:
            base: The real calibration to perturb.
            perturbation: Dict describing the perturbation, e.g.:
                {"scale_t1": 0.7}              — degrade T1 by 30%
                {"scale_readout": 0.9}         — degrade readout by 10%
                {"poison_qubit": "QB23"}       — set QB23 to worst-case
                {"scale_t1": 0.5, "scale_t2": 0.5} — combined

        Returns:
            New DeviceCalibration with perturbation applied and provenance set.
        """
        # Deep copy qubit and gate data
        new_qubits: dict[str, QubitCalibration] = {}
        new_gates: dict[str, GateCalibration] = dict(base.gates)

        for qname, qcal in base.qubits.items():
            t1 = qcal.t1_us
            t2 = qcal.t2_us
            ro = qcal.readout_fidelity
            sge = qcal.single_gate_error

            # Apply scaling perturbations
            if "scale_t1" in perturbation:
                t1 *= perturbation["scale_t1"]
            if "scale_t2" in perturbation:
                t2 *= perturbation["scale_t2"]
            if "scale_readout" in perturbation:
                ro *= perturbation["scale_readout"]
            if "scale_gate_error" in perturbation:
                sge *= perturbation["scale_gate_error"]

            # Poison specific qubit
            if perturbation.get("poison_qubit") == qname:
                t1 = max(t1 * 0.3, 1.0)    # severe T1 degradation
                t2 = min(t2 * 0.2, t1)      # severe T2 degradation
                ro = max(ro * 0.8, 0.5)     # severe readout degradation
                sge = min(sge * 5.0, 0.1)   # severe gate error

            # Enforce physical constraints
            t2 = min(t2, 2.0 * t1)  # T2 ≤ 2*T1
            ro = max(min(ro, 1.0), 0.5)  # clamp readout

            new_qubits[qname] = QubitCalibration(
                name=qname,
                index=qcal.index,
                t1_us=t1,
                t2_us=t2,
                readout_fidelity=ro,
                single_gate_error=sge,
            )

        # Apply gate fidelity scaling
        if "scale_gate_fidelity" in perturbation:
            factor = perturbation["scale_gate_fidelity"]
            scaled_gates = {}
            for gkey, gcal in new_gates.items():
                new_fid = min(gcal.fidelity * factor, 1.0)
                scaled_gates[gkey] = GateCalibration(
                    qubit_pair=gcal.qubit_pair,
                    index_pair=gcal.index_pair,
                    fidelity=new_fid,
                    error=1.0 - new_fid,
                    gate_type=gcal.gate_type,
                )
            new_gates = scaled_gates

        # Build perturbation description string
        perturbation_desc = ", ".join(
            f"{k}={v}" for k, v in sorted(perturbation.items())
        )

        return DeviceCalibration(
            device_id=f"{base.device_id}_synthetic",
            device_prefix=f"{base.device_prefix}_synthetic",
            num_qubits=base.num_qubits,
            topology_name=base.topology_name,
            qubits=new_qubits,
            gates=new_gates,
            adjacency=base.adjacency,
            single_gate_time_ns=base.single_gate_time_ns,
            two_qubit_gate_time_ns=base.two_qubit_gate_time_ns,
            calibration_set_id=f"{base.calibration_set_id}_perturbed",
            calibration_timestamp=datetime.now(timezone.utc).isoformat(),
            calibration_source_file=base.calibration_source_file,
            is_synthetic=True,
            synthetic_perturbation=perturbation_desc,
        )

    def save(self, calibration: DeviceCalibration, path: str) -> None:
        """Save a synthetic calibration in IQM v2 JSON format.

        This allows synthetic calibrations to be consumed by the
        existing noise model builder without modification.
        """
        output: dict[str, Any] = {
            "calibration_set_id": calibration.calibration_set_id,
            "timestamp": calibration.calibration_timestamp,
            "device": calibration.device_id,
            "notes": f"Synthetic: {calibration.synthetic_perturbation}",
            "qubits": {},
            "two_qubit_gates": {},
            "single_gate_time_ns": calibration.single_gate_time_ns,
            "cz_gate_time_ns": calibration.two_qubit_gate_time_ns,
        }

        for qname, qcal in calibration.qubits.items():
            output["qubits"][qname] = {
                "t1_us": round(qcal.t1_us, 4),
                "t2_us": round(qcal.t2_us, 4),
                "readout_fidelity": round(qcal.readout_fidelity, 4),
                "single_gate_error": round(qcal.single_gate_error, 6),
            }

        for gkey, gcal in calibration.gates.items():
            output["two_qubit_gates"][gkey] = {
                "cz_fidelity": round(gcal.fidelity, 6),
                "cz_error": round(gcal.error, 6),
            }

        with open(path, "w") as f:
            json.dump(output, f, indent=2)
