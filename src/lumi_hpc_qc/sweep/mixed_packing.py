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

import json
import os
import tempfile
import time
from dataclasses import dataclass, field, asdict
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
        packer = MixedPacker(device_qubits=53, device_cal=device_cal)
        packer.add_experiment("tfim_4q", tfim_circuit, tfim_placements)
        packer.add_experiment("ghz_3q", ghz_circuit, ghz_placements)
        rounds = packer.pack(max_rounds=10)
    """

    def __init__(self, device_qubits: int, device_cal):
        if device_cal is None:
            raise ValueError(
                "device_cal is required for correct edge-overlap checking. "
                "Without it, placements sharing a CZ coupling edge but no "
                "qubits could be packed into the same round, causing crosstalk."
            )
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

    Note:
        This function checks qubit overlap only.  CZ edge overlap
        checking is the caller's responsibility.  Both DSatur (via
        ``MixedPacker``) and ``GlobalPoolPacker`` guarantee no edge
        overlap before calling this function.
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

    DEPRECATED parity path: This function uses Z-basis parity only.
    For QPU mixed submissions with X/Y Hamiltonians, basis rotation
    must be applied at circuit construction time. Raises ValueError
    if any observable contains X or Y terms.

    Args:
        raw_counts: Device-width bitstring counts.
        entries: MixedEntry objects with observables.
        device_qubits: Total device qubit count.

    Returns:
        List of energies, one per entry. None if no observable.
    """
    # Guard: reject observables with non-Z/I terms
    for entry in entries:
        if entry.observable is not None:
            for pauli_label in entry.observable.paulis.to_labels():
                non_iz = set(pauli_label) - {"I", "Z"}
                if non_iz:
                    raise ValueError(
                        f"compute_mixed_energies cannot estimate Pauli terms "
                        f"containing {non_iz} from Z-basis measurements alone. "
                        f"For QPU results, basis rotation must be applied at "
                        f"circuit construction time."
                    )

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
                    if p in ("Z",) and i < len(bits):
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


# ═══════════════════════════════════════════════════════════════════════
# v1.4.0 — Global pool packing (cross-experiment)
#
# PoolTask → GlobalPoolPacker.pack() → PackedBatch → compose_mixed_round
#
# RED-RESP-V140-DESIGN-v1.0 (REVISED) §6
# ORANGE-TO-RED-COMMS-023 §2a (architecture sound), §4 (packing manifest)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PoolTask:
    """One atomic packing unit — a pre-built circuit on specific qubits.

    Each PoolTask is the output of ``prebuild_pool_tasks()`` for one
    (seed, placement, pauli_group) triple.  The circuit already has basis
    rotations baked in.  The packer doesn't know or care what science
    this circuit represents — it only checks qubit/edge overlap.
    """
    task_id: str                              # e.g. "s0_p7_g0"
    circuit: QuantumCircuit                   # fully bound, basis-rotated
    physical_indices: list[int]               # qubits on device
    internal_edges: set[tuple[int, int]]      # CZ edges between physical_indices
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys: seed, placement_id, pauli_group_index,
    #   pauli_group_labels, identity_energy, hamiltonian, topology_name


@dataclass
class PackedBatch:
    """One QPU submission — a list of PoolTasks packed into one composite.

    Maps directly to one MixedRound for composition + demux.
    """
    batch_id: int
    tasks: list[PoolTask]
    qubit_utilization: float          # len(all_used_qubits) / device_qubits
    composite: QuantumCircuit | None = None  # built by compose_mixed_round


class _BatchBuilder:
    """Accumulator for one batch during packing."""

    def __init__(self, batch_id: int, device_qubits: int):
        self.batch_id = batch_id
        self.device_qubits = device_qubits
        self.tasks: list[PoolTask] = []
        self.used_qubits: set[int] = set()
        self.used_edges: set[tuple[int, int]] = set()

    def add(self, task: PoolTask) -> None:
        self.tasks.append(task)
        self.used_qubits.update(task.physical_indices)
        self.used_edges.update(task.internal_edges)

    def finish(self) -> PackedBatch:
        util = len(self.used_qubits) / self.device_qubits if self.device_qubits else 0.0
        return PackedBatch(
            batch_id=self.batch_id,
            tasks=list(self.tasks),
            qubit_utilization=util,
        )


class GlobalPoolPacker:
    """Pack a flat pool of tasks into dense QPU batches.

    Greedy backfill: iterate tasks sorted by qubit count (descending),
    greedily add each task to the first batch where it fits (no qubit
    or edge overlap).  If no batch fits, start a new batch.

    This is first-fit-decreasing bin packing — O(N × B) where N is
    tasks and B is batches.  For 2,640 tasks and ~330 batches, this is
    ~870K comparisons, each a set intersection — sub-second.

    Args:
        device_qubits: Total qubit count (53 for Q50).
        device_cal: DeviceCalibration for edge overlap checking.
        objective: ``"max_throughput"`` (default).  Future:
            ``"capped_utilization"``, ``"single_topology"`` (v1.4.1+).

    Raises:
        ValueError: If ``objective`` is not recognised.

    RED-RESP-V140-DESIGN-v1.0 (REVISED) §5 — tunable objectives.
    """

    _KNOWN_OBJECTIVES = {"max_throughput", "capped_utilization", "single_topology"}

    def __init__(
        self,
        device_qubits: int,
        device_cal: Any,
        *,
        objective: str = "max_throughput",
    ):
        if device_cal is None:
            raise ValueError("device_cal is required")
        if objective not in self._KNOWN_OBJECTIVES:
            raise ValueError(
                f"Unknown packing objective '{objective}'. "
                f"Known: {sorted(self._KNOWN_OBJECTIVES)}"
            )
        if objective != "max_throughput":
            raise ValueError(
                f"Objective '{objective}' is not yet implemented (v1.4.1+). "
                f"Only 'max_throughput' is available in v1.4.0."
            )
        self._device_qubits = device_qubits
        self._device_cal = device_cal
        self._objective = objective

    @property
    def objective(self) -> str:
        return self._objective

    def pack(
        self,
        tasks: list[PoolTask],
        *,
        packing_seed: int = 42,
    ) -> list[PackedBatch]:
        """Pack tasks into batches using greedy backfill.

        Algorithm (max_throughput):
          1. Shuffle pool (deterministic with packing_seed) to break
             submission-order bias for equal-size tasks
          2. Sort by qubit count descending (big tasks first)
          3. For each task, try to add to each existing batch:
             - Check qubit overlap: task ∩ batch == ∅
             - Check edge overlap: task ∩ batch == ∅
          4. If no batch fits, create a new batch

        Deterministic given the same input + packing_seed.

        Args:
            tasks: Flat list of PoolTasks from all experiment groups.
            packing_seed: Random seed for shuffle before sort.

        Returns:
            List of PackedBatch, ordered by batch_id.
        """
        import random
        rng = random.Random(packing_seed)

        pool = list(tasks)
        rng.shuffle(pool)
        pool.sort(key=lambda t: len(t.physical_indices), reverse=True)

        batches: list[_BatchBuilder] = []

        for task in pool:
            task_qubits = set(task.physical_indices)
            task_edges = task.internal_edges
            placed = False

            for batch in batches:
                if (not (task_qubits & batch.used_qubits)
                        and not (task_edges & batch.used_edges)):
                    batch.add(task)
                    placed = True
                    break

            if not placed:
                b = _BatchBuilder(len(batches), self._device_qubits)
                b.add(task)
                batches.append(b)

        return [b.finish() for b in batches]


def validate_packed_batch(batch: PackedBatch) -> list[str]:
    """Validate a single packed batch for the 3 per-batch invariants.

    Invariant 1: No qubit overlap between tasks.
    Invariant 2: No edge overlap between tasks.
    Invariant 3: No duplicate task IDs.

    Returns:
        List of error strings.  Empty = valid.
    """
    errors: list[str] = []
    all_qubits: set[int] = set()
    all_edges: set[tuple[int, int]] = set()
    task_ids: set[str] = set()

    for task in batch.tasks:
        q = set(task.physical_indices)
        e = task.internal_edges

        overlap = q & all_qubits
        if overlap:
            errors.append(
                f"Qubit overlap: {task.task_id} shares qubits {overlap}"
            )

        edge_overlap = e & all_edges
        if edge_overlap:
            errors.append(
                f"Edge overlap: {task.task_id} shares edges {edge_overlap}"
            )

        if task.task_id in task_ids:
            errors.append(f"Duplicate task: {task.task_id}")

        all_qubits |= q
        all_edges |= e
        task_ids.add(task.task_id)

    return errors


def validate_packing(
    batches: list[PackedBatch],
    original_pool_size: int,
) -> list[str]:
    """Validate the full packing result against campaign-level invariants.

    Invariant 3 (global): Every task appears exactly once.
    Invariant 4: Determinism is the caller's responsibility (pack twice,
                 compare).  Not checked here.

    Args:
        batches: Output of ``GlobalPoolPacker.pack()``.
        original_pool_size: ``len(tasks)`` passed to ``pack()``.

    Returns:
        List of error strings.  Empty = valid.
    """
    errors: list[str] = []

    # Per-batch validation
    for batch in batches:
        errors.extend(validate_packed_batch(batch))

    # Global: every task exactly once
    all_ids: list[str] = []
    for batch in batches:
        all_ids.extend(t.task_id for t in batch.tasks)

    if len(all_ids) != original_pool_size:
        errors.append(
            f"Task count mismatch: pool had {original_pool_size}, "
            f"packing produced {len(all_ids)}"
        )
    if len(all_ids) != len(set(all_ids)):
        from collections import Counter
        dupes = [tid for tid, c in Counter(all_ids).items() if c > 1]
        errors.append(f"Duplicate tasks across batches: {dupes}")

    return errors


# ═══════════════════════════════════════════════════════════════════════
# Packing manifest — static record of task→batch assignment
#
# Written once before QPU submission.  Never modified.  Used on resume
# to replay the original batch composition (replay + skip completed).
#
# ORANGE-TO-RED-COMMS-023 §4 — schema approved by Red.
# RED-RESP-V140-DESIGN-v1.0 (REVISED) §3b.
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class PackingManifest:
    """Static record of how tasks were packed into batches.

    Written atomically after ``GlobalPoolPacker.pack()`` completes,
    before any QPU submission.  One file per campaign.  Never modified
    after creation — the campaign manifest tracks completion.

    Resume flow:
      1. ``GlobalPoolPacker.pack()`` → ``PackingManifest`` (written once)
      2. Sweep engine executes batches → ``CampaignManifest`` (updated)
      3. On resume: load both.  Packing manifest = planned.
         Campaign manifest = completed.  Execute the difference.
    """
    packing_version: str = "1.0"
    strategy: str = "global_pool"
    objective: str = "max_throughput"
    packing_seed: int = 0              # set by caller from PackingConfig.seed
    device_qubits: int = 0             # set by caller from backend device info
    total_tasks: int = 0
    total_batches: int = 0
    mean_utilization: float = 0.0
    created_at: str = ""
    batches: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_packed_batches(
        cls,
        batches: list[PackedBatch],
        *,
        strategy: str = "global_pool",
        objective: str = "max_throughput",
        packing_seed: int = 0,
        device_qubits: int = 0,
    ) -> PackingManifest:
        """Build a manifest from the packer output."""
        from datetime import datetime, timezone

        batch_records: list[dict[str, Any]] = []
        total_util = 0.0

        for batch in batches:
            task_records = []
            for task in batch.tasks:
                task_records.append({
                    "task_id": task.task_id,
                    "seed": task.metadata.get("seed"),
                    "placement_id": task.metadata.get("placement_id"),
                    "pauli_group_index": task.metadata.get("pauli_group_index"),
                    "hamiltonian": task.metadata.get("hamiltonian"),
                    "topology_name": task.metadata.get("topology_name"),
                    "physical_indices": task.physical_indices,
                })

            batch_records.append({
                "batch_id": batch.batch_id,
                "qubit_utilization": round(batch.qubit_utilization, 4),
                "n_tasks": len(batch.tasks),
                "tasks": task_records,
            })
            total_util += batch.qubit_utilization

        n_batches = len(batches)
        return cls(
            strategy=strategy,
            objective=objective,
            packing_seed=packing_seed,
            device_qubits=device_qubits,
            total_tasks=sum(len(b.tasks) for b in batches),
            total_batches=n_batches,
            mean_utilization=round(total_util / n_batches, 4) if n_batches else 0.0,
            created_at=datetime.now(timezone.utc).isoformat(),
            batches=batch_records,
        )

    def save(self, path: str | os.PathLike) -> None:
        """Atomic write — temp file + rename.

        Same pattern as ``CampaignManifest.save()`` for crash safety.
        """
        path = os.fspath(path)
        parent = os.path.dirname(path) or "."
        data = asdict(self)

        fd, tmp_path = tempfile.mkstemp(
            dir=parent,
            prefix=".packing_manifest_",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
            os.rename(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    @classmethod
    def load(cls, path: str | os.PathLike) -> PackingManifest:
        """Load an existing packing manifest for resume."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            packing_version=data.get("packing_version", "1.0"),
            strategy=data.get("strategy", "global_pool"),
            objective=data.get("objective", "max_throughput"),
            packing_seed=data.get("packing_seed", 0),
            device_qubits=data.get("device_qubits", 0),
            total_tasks=data.get("total_tasks", 0),
            total_batches=data.get("total_batches", 0),
            mean_utilization=data.get("mean_utilization", 0.0),
            created_at=data.get("created_at", ""),
            batches=data.get("batches", []),
        )

    def completed_batch_ids(self, campaign_manifest_path: str | os.PathLike) -> set[int]:
        """Cross-reference with campaign manifest to find completed batches.

        Loads the campaign manifest (if it exists), finds which batch_ids
        are marked completed, and returns them as a set for the resume
        skip-logic.
        """
        from lumi_hpc_qc.sweep.campaign_manifest import CampaignManifest

        cpath = os.fspath(campaign_manifest_path)
        if not os.path.exists(cpath):
            return set()

        cm = CampaignManifest.load(cpath)
        completed_tasks = set(cm.completed_tasks())

        done: set[int] = set()
        for batch_rec in self.batches:
            batch_task_ids = [t["task_id"] for t in batch_rec["tasks"]]
            if all(tid in completed_tasks for tid in batch_task_ids):
                done.add(batch_rec["batch_id"])
        return done
