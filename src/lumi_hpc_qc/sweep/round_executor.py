# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Round executor — multi-round packed circuit execution.

Executes all packing rounds for a sweep: compose each round into a
composite circuit, run it, demultiplex results, and collect per-placement
energies. Supports both shot-based (QPU/noisy Aer) and exact (statevector/
density_matrix) execution modes.

RED-SPEC-002 §3 — Multi-Round Same-Circuit Packing
RED-SPEC-002 §3.3 — Execution Strategy (auto-select packing vs individual)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from lumi_hpc_qc.sweep.placement_solver import Placement
from lumi_hpc_qc.sweep.circuit_composer import compose_round, compose_round_for_density_matrix
from lumi_hpc_qc.sweep.demultiplexer import (
    compute_placement_energies,
    compute_placement_energies_exact,
)


@dataclass
class PlacementResult:
    """Result for one placement from packed execution."""
    placement_id: str = ""
    physical_indices: list[int] = field(default_factory=list)
    qubit_names: list[str] = field(default_factory=list)
    topology_hash: str = ""
    energy: float | None = None
    round_index: int = 0
    execution_time_s: float = 0.0
    error: str | None = None


@dataclass
class RoundResult:
    """Result of executing one packing round."""
    round_index: int = 0
    num_placements: int = 0
    execution_time_s: float = 0.0
    placement_results: list[PlacementResult] = field(default_factory=list)
    error: str | None = None


@dataclass
class MultiRoundResult:
    """Aggregate result of all packing rounds."""
    total_rounds: int = 0
    total_placements: int = 0
    total_time_s: float = 0.0
    rounds: list[RoundResult] = field(default_factory=list)
    placement_results: list[PlacementResult] = field(default_factory=list)


def execute_packed_rounds(
    circuit: Any,
    observable: Any,
    rounds: list[list[Placement]],
    device_qubits: int,
    *,
    method: str = "density_matrix",
    shots: int = 4096,
    seed: int = 42,
    noise_model: Any = None,
    device: str = "CPU",
    params: Any = None,
) -> MultiRoundResult:
    """Execute all packing rounds with multiplexed circuits.

    For each round, composes placements into a single device-width
    circuit, executes once, and demultiplexes per-placement results.

    Args:
        circuit: Template circuit (N logical qubits, no measurements).
        observable: SparsePauliOp for energy computation.
        rounds: List of rounds, each a list of non-overlapping Placements.
        device_qubits: Total device qubit count (e.g., 53 for Q50).
        method: "density_matrix" or "statevector".
        shots: Measurement shots (0 for exact).
        seed: Base seed (incremented per round).
        noise_model: Optional Aer noise model.
        device: "CPU" or "GPU".
        params: Optional parameter values for parameterized circuits.

    Returns:
        MultiRoundResult with per-placement energies.
    """
    from qiskit_aer import AerSimulator

    result = MultiRoundResult(total_rounds=len(rounds))
    all_placement_results: list[PlacementResult] = []
    t_total_start = time.time()

    for round_idx, round_placements in enumerate(rounds):
        round_seed = seed + round_idx * 1000
        t_round_start = time.time()
        rr = RoundResult(
            round_index=round_idx,
            num_placements=len(round_placements),
        )

        try:
            if shots == 0 or method == "statevector":
                # Exact evaluation via density matrix
                energies = _execute_round_exact(
                    circuit, observable, round_placements, device_qubits,
                    method=method, seed=round_seed, noise_model=noise_model,
                    device=device, params=params,
                )
            else:
                # Shot-based evaluation
                energies = _execute_round_shots(
                    circuit, observable, round_placements, device_qubits,
                    shots=shots, seed=round_seed, noise_model=noise_model,
                    device=device, params=params,
                )

            rr.execution_time_s = time.time() - t_round_start

            for p_idx, (placement, energy) in enumerate(zip(round_placements, energies)):
                pr = PlacementResult(
                    placement_id=placement.placement_id
                                 if hasattr(placement, 'placement_id')
                                 else f"r{round_idx}_p{p_idx}",
                    physical_indices=list(placement.physical_indices),
                    qubit_names=list(placement.qubit_mapping.values())
                                if placement.qubit_mapping else [],
                    topology_hash=placement.topology_hash,
                    energy=energy,
                    round_index=round_idx,
                    execution_time_s=rr.execution_time_s / len(round_placements),
                )
                rr.placement_results.append(pr)
                all_placement_results.append(pr)

        except Exception as e:
            rr.error = str(e)
            rr.execution_time_s = time.time() - t_round_start
            # Mark all placements in this round as failed
            for p_idx, placement in enumerate(round_placements):
                pr = PlacementResult(
                    placement_id=f"r{round_idx}_p{p_idx}",
                    physical_indices=list(placement.physical_indices),
                    topology_hash=placement.topology_hash,
                    round_index=round_idx,
                    error=str(e),
                )
                rr.placement_results.append(pr)
                all_placement_results.append(pr)

        result.rounds.append(rr)

    result.total_time_s = time.time() - t_total_start
    result.total_placements = len(all_placement_results)
    result.placement_results = all_placement_results
    return result


def _execute_round_exact(
    circuit, observable, placements, device_qubits,
    *, method, seed, noise_model, device, params,
) -> list[float]:
    """Execute one round using exact density matrix evaluation."""
    from qiskit_aer import AerSimulator

    composite = compose_round_for_density_matrix(
        circuit, placements, device_qubits, params=params,
    )

    sim_opts = {"method": "density_matrix", "device": device}
    if noise_model is not None:
        sim_opts["noise_model"] = noise_model
    sim = AerSimulator(**sim_opts)

    job = sim.run(composite, shots=0, seed_simulator=seed)
    result = job.result()
    dm = np.array(result.data()["density_matrix"])

    return compute_placement_energies_exact(
        dm, placements, observable, device_qubits,
    )


def _execute_round_shots(
    circuit, observable, placements, device_qubits,
    *, shots, seed, noise_model, device, params,
) -> list[float]:
    """Execute one round using shot-based measurement."""
    from qiskit_aer import AerSimulator

    composite = compose_round(
        circuit, placements, device_qubits, params=params,
    )

    sim_opts = {"method": "density_matrix", "device": device}
    if noise_model is not None:
        sim_opts["noise_model"] = noise_model
    sim = AerSimulator(**sim_opts)

    job = sim.run(composite, shots=shots, seed_simulator=seed)
    result = job.result()
    raw_counts = result.get_counts()

    return compute_placement_energies(
        raw_counts, placements, observable, device_qubits,
    )
