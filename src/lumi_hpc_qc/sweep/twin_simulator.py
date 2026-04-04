# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Twin simulator — multi-calibration noise battery per placement.

For each (placement, calibration) pair, runs all 11 noise environments
and writes results to HDF5. Handles noiseless deduplication across
calibrations (environments 1-2 depend only on topology, not calibration).

RED-SPEC-002 §4 — Multi-Calibration Twin Simulation Battery
RED-DIRECTIVE-E4-SCHEMA-v1.0 §6

Integration:
    E1 (placement solver) → placement + per-qubit calibration
    E4 (this) → builds noise model per env, dispatches execution
    E2 (execution planner) → routes to CPU or GPU
    E5 (eval runner) → executes circuit, returns energy
    E3 (HDF5 writer) → writes result with WAL safety

CRITICAL: Noiseless deduplication
    - noiseless and topology_noiseless depend only on placement topology
    - Computed once per unique topology hash
    - Stored as HDF5 soft links across calibration groups
    - Saves ~10% of simulation budget
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lumi_hpc_qc.sweep.noise_configs import (
    NOISE_ENVIRONMENTS,
    NOISE_ENV_BY_NAME,
    NOISELESS_ENVS,
    NOISY_ENVS,
    NoiseConfig,
    get_active_channels_string,
)


@dataclass
class TwinResult:
    """Result of one (placement × calibration × environment) simulation.

    Attributes:
        placement_id: Identifier for the physical qubit placement.
        calibration_id: Identifier for the calibration source.
        environment: Name of the noise environment.
        energy: Computed expectation value (None if failed).
        counts: Raw measurement counts (None for statevector).
        execution_time_s: Wall time for this simulation.
        backend_used: "aer_cpu" or "aer_gpu".
        noise_channels_active: Human-readable active channels string.
        measurement_stats_interval: Stats capture interval used.
        tier: Noise tier ("noiseless", "A", "B", "full").
        is_deduplicated: True if this result was soft-linked (not recomputed).
        error: Error message if failed.
        metadata: Additional metadata.
    """
    placement_id: str = ""
    calibration_id: str = ""
    environment: str = ""
    energy: float | None = None
    counts: dict[str, int] | None = None
    execution_time_s: float = 0.0
    backend_used: str = ""
    noise_channels_active: str = ""
    measurement_stats_interval: int = 0
    tier: str = ""
    is_deduplicated: bool = False
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PlacementBatteryResult:
    """All 11 environment results for one (placement × calibration)."""
    placement_id: str = ""
    calibration_id: str = ""
    topology_hash: str = ""
    results: list[TwinResult] = field(default_factory=list)
    total_time_s: float = 0.0
    simulated_count: int = 0
    deduplicated_count: int = 0


def build_placement_noise_model(
    calibration_data: dict[str, Any],
    qubit_names: list[str],
    noise_channels: dict[str, bool] | None,
):
    """Build an Aer NoiseModel for specific physical qubits.

    Unlike build_noise_model() which auto-selects qubits by fidelity,
    this function builds a noise model for the exact qubits in a
    placement. The qubit indices in the noise model correspond to
    the order in qubit_names: qubit_names[0] → index 0, etc.

    Args:
        calibration_data: Parsed calibration JSON dict.
        qubit_names: Ordered list of physical qubit names (e.g., ["QB6", "QB7"]).
        noise_channels: Dict of channel → bool. None = noiseless (returns None).

    Returns:
        (noise_model, coupling_map) or (None, None) if noiseless.
    """
    if noise_channels is None:
        return None, None

    # Check if any channel is active
    if not any(noise_channels.values()):
        return None, None

    from qiskit_aer.noise import NoiseModel, depolarizing_error, ReadoutError
    from qiskit_aer.noise.errors.standard_errors import thermal_relaxation_error
    from qiskit.transpiler import CouplingMap

    qubits_data = calibration_data.get("qubits", {})
    gates_data = calibration_data.get("two_qubit_gates", {})
    cz_time_ns = calibration_data.get("cz_gate_time_ns", 100)

    name_to_idx = {name: i for i, name in enumerate(qubit_names)}
    noise_model = NoiseModel()

    # Single-qubit depolarizing
    if noise_channels.get("single_qubit_depolarizing", False):
        for i, qname in enumerate(qubit_names):
            qdata = qubits_data.get(qname, {})
            sg_err = qdata.get("single_gate_error", 0.001)
            if sg_err > 0:
                dep_err = depolarizing_error(sg_err, 1)
                noise_model.add_quantum_error(
                    dep_err, ['rx', 'ry', 'rz', 'x', 'h', 'sx'], [i]
                )

    # Two-qubit depolarizing
    if noise_channels.get("two_qubit_depolarizing", False):
        for gate_pair, gate_data in gates_data.items():
            parts = gate_pair.split("-")
            if len(parts) != 2:
                continue
            q1, q2 = parts
            if q1 not in name_to_idx or q2 not in name_to_idx:
                continue
            i, j = name_to_idx[q1], name_to_idx[q2]
            cz_err = gate_data.get("cz_error", 0.005)
            if cz_err > 0:
                dep_err_2q = depolarizing_error(cz_err, 2)
                noise_model.add_quantum_error(dep_err_2q, ['cx', 'cz'], [i, j])
                noise_model.add_quantum_error(dep_err_2q, ['cx', 'cz'], [j, i])

    # T1/T2 thermal relaxation
    if noise_channels.get("t1_relaxation", False) or noise_channels.get("t2_dephasing", False):
        for i, qname in enumerate(qubit_names):
            qdata = qubits_data.get(qname, {})
            t1_ns = qdata.get("t1_us", 50.0) * 1e3
            t2_ns = qdata.get("t2_us", 20.0) * 1e3

            if not noise_channels.get("t1_relaxation", False):
                t1_ns = float('inf')
            if not noise_channels.get("t2_dephasing", False):
                t2_ns = float('inf')
            if t2_ns > 2 * t1_ns:
                t2_ns = 2 * t1_ns
            if t1_ns == float('inf') and t2_ns == float('inf'):
                continue

            thermal_err = thermal_relaxation_error(t1_ns, t2_ns, cz_time_ns)
            noise_model.add_quantum_error(thermal_err, ['id', 'delay'], [i])

    # Readout error
    if noise_channels.get("readout_error", False):
        for i, qname in enumerate(qubit_names):
            qdata = qubits_data.get(qname, {})
            ro_fid = qdata.get("readout_fidelity", 0.97)
            p_error = (1 - ro_fid) / 2
            ro_err = ReadoutError([
                [1 - p_error, p_error],
                [p_error, 1 - p_error],
            ])
            noise_model.add_readout_error(ro_err, [i])

    # Coupling map from placement edges
    coupling_edges = []
    for gate_pair in gates_data:
        parts = gate_pair.split("-")
        if len(parts) != 2:
            continue
        q1, q2 = parts
        if q1 in name_to_idx and q2 in name_to_idx:
            i, j = name_to_idx[q1], name_to_idx[q2]
            coupling_edges.append([i, j])
            coupling_edges.append([j, i])

    coupling_map = CouplingMap(coupling_edges) if coupling_edges else None
    return noise_model, coupling_map


def run_twin_battery(
    circuit: Any,
    observable: Any,
    qubit_names: list[str],
    calibration_data: dict[str, Any],
    calibration_id: str,
    placement_id: str,
    topology_hash: str,
    *,
    environments: list[NoiseConfig] | None = None,
    seed: int = 42,
    device: str = "CPU",
    noiseless_cache: dict[str, TwinResult] | None = None,
) -> PlacementBatteryResult:
    """Run the full 11-environment twin simulation battery.

    Args:
        circuit: QuantumCircuit (unparameterized, no measurements).
        observable: SparsePauliOp for expectation value computation.
        qubit_names: Physical qubit names for this placement.
        calibration_data: Parsed calibration JSON dict.
        calibration_id: Identifier for this calibration source.
        placement_id: Identifier for this placement.
        topology_hash: Abstract topology hash (for deduplication key).
        environments: List of NoiseConfigs to run. Default: all 11.
        seed: Base seed (incremented per environment for independence).
        device: "CPU" or "GPU".
        noiseless_cache: Dict of topology_hash → TwinResult for deduplication.
            Pass a shared dict across calibrations to enable deduplication.

    Returns:
        PlacementBatteryResult with all environment results.
    """
    from lumi_hpc_qc.sweep.eval_runner import evaluate_circuit, EvalResult
    from lumi_hpc_qc.sweep.circuit_loader import LoadedCircuit

    if environments is None:
        environments = NOISE_ENVIRONMENTS

    if noiseless_cache is None:
        noiseless_cache = {}

    battery = PlacementBatteryResult(
        placement_id=placement_id,
        calibration_id=calibration_id,
        topology_hash=topology_hash,
    )

    t_total_start = time.time()

    # Wrap circuit for eval_runner
    loaded = LoadedCircuit(
        circuit=circuit,
        num_qubits=circuit.num_qubits,
        num_parameters=0,
        is_parameterized=False,
        connectivity=[],  # Not needed for execution
        source=f"twin:{placement_id}",
    )

    for env_idx, env in enumerate(environments):
        env_seed = seed + env_idx

        # ── Noiseless deduplication ──
        if env.tier == "noiseless":
            obs_hash = hashlib.sha256(str(observable).encode()).hexdigest()[:12]
            cache_key = f"{obs_hash}:{circuit.num_qubits}:{topology_hash}:{env.name}"
            if cache_key in noiseless_cache:
                cached = noiseless_cache[cache_key]
                result = TwinResult(
                    placement_id=placement_id,
                    calibration_id=calibration_id,
                    environment=env.name,
                    energy=cached.energy,
                    counts=cached.counts,
                    execution_time_s=0.0,
                    backend_used=cached.backend_used,
                    noise_channels_active="none",
                    measurement_stats_interval=0,
                    tier=env.tier,
                    is_deduplicated=True,
                )
                battery.results.append(result)
                battery.deduplicated_count += 1
                continue

        # ── Build noise model for this environment ──
        noise_model = None
        coupling_map = None
        if env.channels is not None:
            noise_model, coupling_map = build_placement_noise_model(
                calibration_data, qubit_names, env.channels,
            )
        elif env.coupling_map_source == "calibration":
            # topology_noiseless: no noise, but use placement coupling map
            _, coupling_map = build_placement_noise_model(
                calibration_data, qubit_names,
                {"single_qubit_depolarizing": False, "two_qubit_depolarizing": False,
                 "t1_relaxation": False, "t2_dephasing": False, "readout_error": False},
            )

        # ── Execute ──
        t0 = time.time()
        try:
            eval_result = evaluate_circuit(
                loaded,
                observable=observable,
                method=env.method,
                shots=env.shots,
                seed=env_seed,
                noise_model=noise_model,
                coupling_map=coupling_map,
                device=device,
            )

            result = TwinResult(
                placement_id=placement_id,
                calibration_id=calibration_id,
                environment=env.name,
                energy=eval_result.energy,
                counts=eval_result.counts,
                execution_time_s=eval_result.execution_time_s,
                backend_used=eval_result.backend_used,
                noise_channels_active=get_active_channels_string(env),
                measurement_stats_interval=env.measurement_stats_interval,
                tier=env.tier,
                is_deduplicated=False,
                error=eval_result.error,
            )
        except Exception as e:
            result = TwinResult(
                placement_id=placement_id,
                calibration_id=calibration_id,
                environment=env.name,
                execution_time_s=time.time() - t0,
                noise_channels_active=get_active_channels_string(env),
                measurement_stats_interval=env.measurement_stats_interval,
                tier=env.tier,
                error=str(e),
            )

        battery.results.append(result)
        battery.simulated_count += 1

        # Cache noiseless results for deduplication
        if env.tier == "noiseless" and result.error is None:
            obs_hash = hashlib.sha256(str(observable).encode()).hexdigest()[:12]
            cache_key = f"{obs_hash}:{circuit.num_qubits}:{topology_hash}:{env.name}"
            noiseless_cache[cache_key] = result

    battery.total_time_s = time.time() - t_total_start
    return battery


def run_multi_calibration_battery(
    circuit: Any,
    observable: Any,
    qubit_names: list[str],
    calibrations: list[tuple[str, dict[str, Any]]],
    placement_id: str,
    topology_hash: str,
    *,
    environments: list[NoiseConfig] | None = None,
    seed: int = 42,
    device: str = "CPU",
) -> list[PlacementBatteryResult]:
    """Run the twin battery across multiple calibrations for one placement.

    Handles noiseless deduplication across calibrations automatically.
    The first calibration computes noiseless results; subsequent
    calibrations reuse them via the shared cache.

    Args:
        circuit: Fixed QuantumCircuit.
        observable: SparsePauliOp.
        qubit_names: Physical qubit names for this placement.
        calibrations: List of (calibration_id, calibration_data) pairs.
        placement_id: Placement identifier.
        topology_hash: For deduplication key.
        environments: Noise configs. Default: all 11.
        seed: Base seed.
        device: "CPU" or "GPU".

    Returns:
        List of PlacementBatteryResult, one per calibration.
    """
    noiseless_cache: dict[str, TwinResult] = {}
    results = []

    for cal_id, cal_data in calibrations:
        battery = run_twin_battery(
            circuit=circuit,
            observable=observable,
            qubit_names=qubit_names,
            calibration_data=cal_data,
            calibration_id=cal_id,
            placement_id=placement_id,
            topology_hash=topology_hash,
            environments=environments,
            seed=seed,
            device=device,
            noiseless_cache=noiseless_cache,
        )
        results.append(battery)

    return results
