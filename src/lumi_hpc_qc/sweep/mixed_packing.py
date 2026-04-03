# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""E6b mixed-experiment packing — different circuits share QPU submissions.

Extends E6a's same-circuit packing to support heterogeneous circuits from
different experiments packed into a single device-width submission. Each
experiment's circuit runs on non-overlapping physical qubits within the
same QPU shot.

Key differences from E6a:
  - E6a: one circuit template × N placements → one composite
  - E6b: M different circuits × 1 placement each → one composite

Components:
  - MixedPacker: finds non-overlapping (circuit, placement) combos across experiments
  - compose_mixed_round: builds device-width circuit from heterogeneous entries
  - demux_mixed_counts: routes composite results back to each experiment
  - execute_mixed_round: compose → execute → demux in one call

RED-SPEC-002 §15 — Mixed-Experiment Packing
VE20: Two different circuits packed, demuxed, results match independent runs
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qiskit import QuantumCircuit

from lumi_hpc_qc.sweep.placement_solver import Placement


# ═══════════════════════════════════════════════════════════════════════
# Data types
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class MixedEntry:
    """One experiment's circuit + placement in a mixed submission.

    Each entry represents a single (experiment, circuit, placement) tuple
    that will be composed into the device-width circuit alongside entries
    from other experiments.
    """
    experiment_id: str
    circuit: QuantumCircuit
    placement: Placement
    observable: Any | None = None          # SparsePauliOp for energy computation
    params: Any | None = None              # parameter values (for parameterized circuits)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MixedRound:
    """One mixed-experiment QPU submission round.

    Contains entries from multiple experiments, verified non-overlapping.
    """
    round_id: int
    entries: list[MixedEntry]
    device_qubits: int
    total_qubits_used: int = 0
    co_submitted: list[str] = field(default_factory=list)  # experiment IDs

    def __post_init__(self):
        used = set()
        for e in self.entries:
            used.update(e.placement.physical_indices)
        self.total_qubits_used = len(used)
        self.co_submitted = [e.experiment_id for e in self.entries]


@dataclass
class MixedResult:
    """Result for one experiment's entry from a mixed submission."""
    experiment_id: str
    placement_id: int | str
    energy: float | None = None
    counts: dict[str, int] | None = None
    co_submitted_with: list[str] = field(default_factory=list)
    execution_time_s: float = 0.0
    error: str | None = None


@dataclass
class MixedRoundResult:
    """Results from executing one mixed round."""
    round_id: int
    results: list[MixedResult] = field(default_factory=list)
    total_time_s: float = 0.0
    error: str | None = None


# ═══════════════════════════════════════════════════════════════════════
# Mixed packer — find non-overlapping (circuit, placement) combinations
# ═══════════════════════════════════════════════════════════════════════

class MixedPacker:
    """Pack circuits from different experiments into shared QPU rounds.

    Takes a list of experiment queues — each queue is a (circuit, [placements])
    pair. Finds combinations where one placement from each experiment fits
    non-overlappingly on the device.

    Usage:
        packer = MixedPacker(device_qubits=53)
        packer.add_experiment("tfim_4q", tfim_circuit, tfim_placements)
        packer.add_experiment("ghz_3q", ghz_circuit, ghz_placements)
        rounds = packer.pack(max_rounds=10)
    """

    def __init__(self, device_qubits: int, device_cal: Any = None):
        self._device_qubits = device_qubits
        self._device_cal = device_cal
        self._experiments: list[tuple[str, QuantumCircuit, list[Placement], Any]] = []

    def add_experiment(
        self,
        experiment_id: str,
        circuit: QuantumCircuit,
        placements: list[Placement],
        observable: Any | None = None,
    ) -> None:
        """Add an experiment's circuit and candidate placements."""
        self._experiments.append((experiment_id, circuit, placements, observable))

    def pack(
        self,
        *,
        max_rounds: int | None = None,
        packing_seed: int = 42,
    ) -> list[MixedRound]:
        """Find non-overlapping mixed rounds across experiments.

        For each round, greedily selects one placement per experiment
        such that no physical qubits or coupling edges overlap.

        Args:
            max_rounds: Cap on rounds. None = pack until all experiments
                       have at least one placement packed.
            packing_seed: Seed for deterministic placement ordering.

        Returns:
            List of MixedRound objects.
        """
        import random
        rng = random.Random(packing_seed)

        if not self._experiments:
            return []

        # Build per-experiment placement queues (shuffled, then sorted by score)
        queues: list[tuple[str, QuantumCircuit, list[Placement], Any]] = []
        for exp_id, circuit, placements, obs in self._experiments:
            remaining = list(placements)
            rng.shuffle(remaining)
            remaining.sort(key=lambda p: p.score, reverse=True)
            queues.append((exp_id, circuit, remaining, obs))

        rounds: list[MixedRound] = []
        round_counter = 0
        max_iter = max_rounds if max_rounds else len(queues) * 100

        while round_counter < max_iter:
            # Try to build a round with one entry from each experiment
            used_qubits: set[int] = set()
            used_edges: set[tuple[int, int]] = set()
            entries: list[MixedEntry] = []
            consumed_indices: list[tuple[int, int]] = []  # (queue_idx, placement_idx)

            for q_idx, (exp_id, circuit, remaining, obs) in enumerate(queues):
                if not remaining:
                    continue

                # Find first placement that doesn't overlap
                for p_idx, p in enumerate(remaining):
                    p_qubits = set(p.physical_indices)

                    # Check qubit overlap
                    if p_qubits & used_qubits:
                        continue

                    # Check edge overlap
                    p_edges = set()
                    if self._device_cal is not None:
                        for qi in p.physical_indices:
                            for qj in self._device_cal.adjacency.get(qi, set()):
                                if qj in p_qubits:
                                    p_edges.add((min(qi, qj), max(qi, qj)))

                    if p_edges & used_edges:
                        continue

                    # No overlap — add to this round
                    entries.append(MixedEntry(
                        experiment_id=exp_id,
                        circuit=circuit,
                        placement=p,
                        observable=obs,
                    ))
                    used_qubits |= p_qubits
                    used_edges |= p_edges
                    consumed_indices.append((q_idx, p_idx))
                    break

            if len(entries) < 2:
                # Need at least 2 experiments to be "mixed"
                break

            # Remove consumed placements from queues
            for q_idx, p_idx in sorted(consumed_indices, reverse=True):
                queues[q_idx][2].pop(p_idx)

            rounds.append(MixedRound(
                round_id=round_counter,
                entries=entries,
                device_qubits=self._device_qubits,
            ))
            round_counter += 1

        return rounds


# ═══════════════════════════════════════════════════════════════════════
# Mixed composer — heterogeneous circuits into one device-width circuit
# ═══════════════════════════════════════════════════════════════════════

def compose_mixed_round(
    entries: list[MixedEntry],
    device_qubits: int,
    *,
    add_measurements: bool = True,
    add_barriers: bool = True,
) -> QuantumCircuit:
    """Build a device-width circuit from heterogeneous experiment entries.

    Unlike compose_round (E6a) which takes one circuit template,
    this takes multiple different circuits each with their own placement.

    Args:
        entries: List of MixedEntry, each with its own circuit + placement.
        device_qubits: Total device qubit count.
        add_measurements: If True, add measurements on all used qubits.
        add_barriers: If True, add barriers between experiments.

    Returns:
        A single QuantumCircuit on device_qubits qubits.

    Raises:
        ValueError: If any entries overlap in physical qubits.
    """
    # Verify non-overlapping
    all_used: set[int] = set()
    for entry in entries:
        qubit_set = set(entry.placement.physical_indices)
        overlap = all_used & qubit_set
        if overlap:
            raise ValueError(
                f"Experiment {entry.experiment_id} overlaps on qubits {overlap}"
            )
        all_used.update(qubit_set)

    # Build composite
    n_clbits = device_qubits if add_measurements else 0
    composite = QuantumCircuit(device_qubits, n_clbits)

    for e_idx, entry in enumerate(entries):
        circuit = entry.circuit
        phys = entry.placement.physical_indices
        num_logical = circuit.num_qubits

        if len(phys) != num_logical:
            raise ValueError(
                f"Experiment {entry.experiment_id}: placement has {len(phys)} "
                f"qubits but circuit has {num_logical}"
            )

        # Bind parameters if provided
        if entry.params is not None and circuit.num_parameters > 0:
            param_dict = dict(zip(circuit.parameters, entry.params))
            bound = circuit.assign_parameters(param_dict)
        else:
            bound = circuit

        # Map logical → physical
        qubit_map = {i: phys[i] for i in range(num_logical)}
        for instruction in bound.data:
            op = instruction.operation
            if op.name == "measure" or op.name.startswith("save_"):
                continue

            mapped_qubits = [
                composite.qubits[qubit_map[bound.find_bit(q).index]]
                for q in instruction.qubits
            ]
            composite.append(op, mapped_qubits, [])

        if add_barriers and e_idx < len(entries) - 1:
            composite.barrier()

    if add_measurements:
        for q in sorted(all_used):
            composite.measure(q, q)

    return composite


# ═══════════════════════════════════════════════════════════════════════
# Mixed demultiplexer — route results back to each experiment
# ═══════════════════════════════════════════════════════════════════════

def demux_mixed_counts(
    raw_counts: dict[str, int],
    entries: list[MixedEntry],
    device_qubits: int,
) -> list[dict[str, int]]:
    """Extract per-experiment count distributions from a mixed submission.

    Each experiment may have a different number of qubits. The extraction
    uses each entry's placement.physical_indices to select the relevant
    bits from the device-width bitstring.

    Args:
        raw_counts: Full device bitstring → count dict.
        entries: The MixedEntry objects in the composite.
        device_qubits: Total device qubit count.

    Returns:
        List of per-experiment count dicts, same order as entries.
    """
    per_entry: list[dict[str, int]] = [{} for _ in entries]

    for bitstring, count in raw_counts.items():
        bits = bitstring.replace(" ", "").zfill(device_qubits)

        for e_idx, entry in enumerate(entries):
            phys = entry.placement.physical_indices
            num_logical = entry.circuit.num_qubits

            sub_bits = ""
            for logical_q in range(num_logical):
                phys_q = phys[logical_q]
                bit_pos = device_qubits - 1 - phys_q
                sub_bits += bits[bit_pos]

            if sub_bits in per_entry[e_idx]:
                per_entry[e_idx][sub_bits] += count
            else:
                per_entry[e_idx][sub_bits] = count

    return per_entry


def compute_mixed_energies(
    raw_counts: dict[str, int],
    entries: list[MixedEntry],
    device_qubits: int,
) -> list[float | None]:
    """Compute per-experiment energies from mixed submission counts.

    Args:
        raw_counts: Device-width bitstring counts.
        entries: MixedEntry objects with observables.
        device_qubits: Total device qubit count.

    Returns:
        List of energies, one per entry. None if no observable.
    """
    per_entry_counts = demux_mixed_counts(raw_counts, entries, device_qubits)
    energies: list[float | None] = []

    for e_idx, entry in enumerate(entries):
        if entry.observable is None:
            energies.append(None)
            continue

        counts = per_entry_counts[e_idx]
        total_shots = sum(counts.values())
        if total_shots == 0:
            energies.append(None)
            continue

        energy = 0.0
        for pauli_label, coeff in zip(
            entry.observable.paulis.to_labels(),
            entry.observable.coeffs,
        ):
            parity_sum = 0.0
            for bitstring, count in counts.items():
                bits = bitstring[::-1]
                parity = 0
                for i, p in enumerate(pauli_label[::-1]):
                    if p in ("Z", "Y") and i < len(bits):
                        if bits[i] == "1":
                            parity ^= 1
                sign = 1 - 2 * parity
                parity_sum += sign * count
            energy += float(np.real(coeff)) * (parity_sum / total_shots)

        energies.append(energy)

    return energies


# ═══════════════════════════════════════════════════════════════════════
# End-to-end mixed execution
# ═══════════════════════════════════════════════════════════════════════

def execute_mixed_round(
    mixed_round: MixedRound,
    *,
    method: str = "density_matrix",
    shots: int = 4096,
    seed: int = 42,
    noise_model: Any = None,
    device: str = "CPU",
) -> MixedRoundResult:
    """Execute a mixed-experiment round: compose → run → demux.

    Args:
        mixed_round: The MixedRound to execute.
        method: Simulation method.
        shots: Measurement shots (must be > 0 for mixed packing).
        seed: Simulator seed.
        noise_model: Optional Aer noise model.
        device: "CPU" or "GPU".

    Returns:
        MixedRoundResult with per-experiment results.
    """
    from qiskit_aer import AerSimulator

    t_start = time.time()
    round_result = MixedRoundResult(round_id=mixed_round.round_id)

    try:
        # Compose
        composite = compose_mixed_round(
            mixed_round.entries,
            mixed_round.device_qubits,
            add_measurements=True,
            add_barriers=True,
        )

        # Execute
        sim_opts = {"method": method, "device": device}
        if noise_model is not None:
            sim_opts["noise_model"] = noise_model
        sim = AerSimulator(**sim_opts)

        job = sim.run(composite, shots=shots, seed_simulator=seed)
        result = job.result()
        raw_counts = result.get_counts()

        # Demux
        energies = compute_mixed_energies(
            raw_counts, mixed_round.entries, mixed_round.device_qubits,
        )
        per_entry_counts = demux_mixed_counts(
            raw_counts, mixed_round.entries, mixed_round.device_qubits,
        )

        co_ids = [e.experiment_id for e in mixed_round.entries]

        for e_idx, entry in enumerate(mixed_round.entries):
            others = [eid for eid in co_ids if eid != entry.experiment_id]
            round_result.results.append(MixedResult(
                experiment_id=entry.experiment_id,
                placement_id=entry.placement.placement_id,
                energy=energies[e_idx],
                counts=per_entry_counts[e_idx],
                co_submitted_with=others,
                execution_time_s=(time.time() - t_start) / len(mixed_round.entries),
            ))

    except Exception as e:
        round_result.error = str(e)
        for entry in mixed_round.entries:
            round_result.results.append(MixedResult(
                experiment_id=entry.experiment_id,
                placement_id=entry.placement.placement_id,
                error=str(e),
            ))

    round_result.total_time_s = time.time() - t_start
    return round_result
