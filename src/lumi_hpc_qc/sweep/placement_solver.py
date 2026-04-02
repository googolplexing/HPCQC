# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Generalized placement solver for Phase E sweep engine.

Replaces Phase C's fixed-size solver with a general subgraph isomorphism
solver using rustworkx VF2++. Finds every valid physical qubit placement
for arbitrary circuits on arbitrary QPU topologies, then packs them into
multi-round non-overlapping execution batches.

Key capabilities:
  - Subgraph isomorphism via rustworkx vf2_mapping (not brute-force DFS)
  - Multi-device: runs against any DeviceCalibration from calibration adapters
  - Multi-round packing: deterministic non-overlapping batches with packing_seed
  - Placement scoring: max_fidelity, max_connectivity, min_error, diverse
  - Topology equivalence tagging: cross-device placement matching for ML
  - Lazy device loading: calibrations loaded on first access

Phase E — RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0, System 1
Supersedes: plugins/placement/solver.py (Phase C — deprecated, not removed)
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from typing import Any

import rustworkx as rx

from lumi_hpc_qc.plugins.calibration_adapters.base import (
    DeviceCalibration,
)


@dataclass
class Placement:
    """A single valid physical qubit placement for a circuit.

    Attributes:
        placement_id: Unique within this solver run.
        device_id: Which device this placement is on.
        device_prefix: HDF5 naming prefix (e.g., 'vtt_q50').
        qubit_mapping: {logical_qubit_index: physical_qubit_name}
        physical_indices: Physical qubit indices on the device.
        score: Strategy-dependent quality score.
        internal_edges: Number of coupling edges within the placement.
        avg_readout_fidelity: Mean readout fidelity across placement qubits.
        avg_gate_fidelity: Mean CZ gate fidelity across internal edges.
        topology_hash: Hash of the local subgraph topology for equivalence.
        per_qubit_calibration: Per-qubit T1/T2/readout for ML features.
    """

    placement_id: int
    device_id: str
    device_prefix: str
    qubit_mapping: dict[int, str]
    physical_indices: list[int]
    score: float
    internal_edges: int
    avg_readout_fidelity: float
    avg_gate_fidelity: float
    topology_hash: str = ""
    per_qubit_calibration: dict[str, dict[str, float]] = field(
        default_factory=dict
    )


@dataclass
class PackingRound:
    """One QPU submission batch — non-overlapping placements."""

    round_id: int
    placements: list[Placement]
    total_qubits_used: int
    device_id: str


class GeneralPlacementSolver:
    """Find all valid placements for arbitrary circuits on arbitrary QPUs.

    Usage:
        solver = GeneralPlacementSolver()
        solver.add_device(calibration)

        placements = solver.find_all_placements(
            circuit_edges=[(0,1), (1,2), (2,3)],
            circuit_qubits=4,
        )

        rounds = solver.pack_rounds(placements, packing_seed=42)
    """

    def __init__(self) -> None:
        self._devices: dict[str, DeviceCalibration] = {}
        self._device_graphs: dict[str, rx.PyGraph] = {}
        self._placement_counter = 0

    def add_device(self, calibration: DeviceCalibration) -> None:
        """Register a device for placement search."""
        self._devices[calibration.device_id] = calibration

    def _get_device_graph(self, device_id: str) -> rx.PyGraph:
        """Build or return cached rustworkx graph for a device."""
        if device_id not in self._device_graphs:
            cal = self._devices[device_id]
            graph = rx.PyGraph()

            # Add qubit nodes — node data = qubit index
            idx_map: dict[int, int] = {}
            for qname in sorted(cal.qubits.keys()):
                qcal = cal.qubits[qname]
                node_idx = graph.add_node(qcal.index)
                idx_map[qcal.index] = node_idx

            # Add coupling edges
            added_edges: set[tuple[int, int]] = set()
            for gate_cal in cal.gates.values():
                i, j = gate_cal.index_pair
                edge = (min(i, j), max(i, j))
                if edge not in added_edges and i in idx_map and j in idx_map:
                    graph.add_edge(idx_map[i], idx_map[j], None)
                    added_edges.add(edge)

            self._device_graphs[device_id] = graph

        return self._device_graphs[device_id]

    def find_all_placements(
        self,
        circuit_edges: list[tuple[int, int]],
        circuit_qubits: int,
        device_ids: list[str] | None = None,
        strategy: str = "max_fidelity",
        max_placements: int | None = None,
        call_limit: int = 100_000,
    ) -> list[Placement]:
        """Find all valid placements across specified devices.

        Args:
            circuit_edges: Connectivity as (q_i, q_j) pairs.
                For a linear 4q chain: [(0,1), (1,2), (2,3)].
            circuit_qubits: Number of qubits in the circuit.
            device_ids: Devices to search. None = all registered.
            strategy: Scoring strategy for ranking.
            max_placements: Cap on total placements (None = all).
            call_limit: VF2 backtracking limit (safety valve).

        Returns:
            List of Placement objects, sorted by score descending.
        """
        targets = device_ids or list(self._devices.keys())
        all_placements: list[Placement] = []

        # Build circuit graph
        circuit_graph = rx.PyGraph()
        for i in range(circuit_qubits):
            circuit_graph.add_node(i)
        added: set[tuple[int, int]] = set()
        for i, j in circuit_edges:
            edge = (min(i, j), max(i, j))
            if edge not in added:
                circuit_graph.add_edge(i, j, None)
                added.add(edge)

        for dev_id in targets:
            if dev_id not in self._devices:
                print(f"  Warning: device '{dev_id}' not registered")
                continue

            cal = self._devices[dev_id]
            device_graph = self._get_device_graph(dev_id)

            # VF2 subgraph isomorphism
            try:
                mappings = list(rx.vf2_mapping(
                    device_graph,
                    circuit_graph,
                    subgraph=True,
                    induced=False,
                    call_limit=call_limit,
                ))
            except Exception as e:
                print(f"  VF2 error on {dev_id}: {e}, falling back to DFS")
                mappings = self._dfs_fallback(
                    cal, circuit_edges, circuit_qubits
                )

            if not mappings:
                print(
                    f"  Device {dev_id}: 0 valid placements for "
                    f"{circuit_qubits}q circuit"
                )
                continue

            # Convert VF2 mappings to Placement objects
            idx_to_name = cal.index_to_qubit_name
            seen: set[frozenset[int]] = set()

            for mapping in mappings:
                # mapping: {device_graph_node: circuit_graph_node}
                # device_graph node data = device qubit index
                phys_indices = []
                qubit_mapping: dict[int, str] = {}
                for dev_node, circ_node in mapping.items():
                    dev_qubit_idx = device_graph[dev_node]
                    phys_indices.append(dev_qubit_idx)
                    qubit_mapping[circ_node] = idx_to_name.get(
                        dev_qubit_idx, f"Q{dev_qubit_idx}"
                    )

                phys_indices.sort()
                qubit_set = frozenset(phys_indices)

                # Deduplicate automorphisms
                if qubit_set in seen:
                    continue
                seen.add(qubit_set)

                score = self._score_placement(phys_indices, cal, strategy)
                ie = self._count_internal_edges(phys_indices, cal)
                avg_ro = self._avg_readout(phys_indices, cal)
                avg_cz = self._avg_gate_fidelity(phys_indices, cal)
                topo_hash = self._topology_hash(phys_indices, cal)
                per_q = self._per_qubit_calibration(phys_indices, cal)

                all_placements.append(Placement(
                    placement_id=self._placement_counter,
                    device_id=dev_id,
                    device_prefix=cal.device_prefix,
                    qubit_mapping=qubit_mapping,
                    physical_indices=phys_indices,
                    score=score,
                    internal_edges=ie,
                    avg_readout_fidelity=avg_ro,
                    avg_gate_fidelity=avg_cz,
                    topology_hash=topo_hash,
                    per_qubit_calibration=per_q,
                ))
                self._placement_counter += 1

            print(
                f"  Device {dev_id}: {len(seen)} valid placements "
                f"for {circuit_qubits}q circuit"
            )

        all_placements.sort(key=lambda p: p.score, reverse=True)
        if max_placements is not None:
            all_placements = all_placements[:max_placements]
        return all_placements

    def pack_rounds(
        self,
        placements: list[Placement],
        packing_seed: int = 42,
    ) -> list[PackingRound]:
        """Pack placements into non-overlapping execution rounds.

        Greedy packing with deterministic ordering from packing_seed.
        Each round has maximum non-overlapping placements. Enforces
        both qubit AND edge non-overlap (required for QPU correctness).
        """
        by_device: dict[str, list[Placement]] = {}
        for p in placements:
            by_device.setdefault(p.device_id, []).append(p)

        all_rounds: list[PackingRound] = []
        round_counter = 0

        for dev_id, dev_placements in by_device.items():
            rng = random.Random(packing_seed)
            remaining = list(dev_placements)
            rng.shuffle(remaining)
            remaining.sort(key=lambda p: p.score, reverse=True)

            while remaining:
                used_qubits: set[int] = set()
                used_edges: set[tuple[int, int]] = set()
                round_placements: list[Placement] = []
                still_remaining = []

                for p in remaining:
                    p_qubits = set(p.physical_indices)
                    if p_qubits & used_qubits:
                        still_remaining.append(p)
                        continue

                    cal = self._devices[dev_id]
                    p_edges = set()
                    for qi in p.physical_indices:
                        for qj in cal.adjacency.get(qi, set()):
                            if qj in p_qubits:
                                p_edges.add((min(qi, qj), max(qi, qj)))

                    if p_edges & used_edges:
                        still_remaining.append(p)
                        continue

                    round_placements.append(p)
                    used_qubits |= p_qubits
                    used_edges |= p_edges

                if round_placements:
                    all_rounds.append(PackingRound(
                        round_id=round_counter,
                        placements=round_placements,
                        total_qubits_used=len(used_qubits),
                        device_id=dev_id,
                    ))
                    round_counter += 1

                remaining = still_remaining

        return all_rounds

    # --- Scoring ---

    def _score_placement(
        self, indices: list[int], cal: DeviceCalibration, strategy: str,
    ) -> float:
        n = len(indices)
        if strategy == "max_fidelity":
            ro = self._avg_readout(indices, cal)
            cz = self._avg_gate_fidelity(indices, cal)
            return (ro + cz) / 2 if cz > 0 else ro
        elif strategy == "max_connectivity":
            edges = self._count_internal_edges(indices, cal)
            max_e = n * (n - 1) / 2
            return edges / max_e if max_e > 0 else 0
        elif strategy == "min_error":
            total_err = 0.0
            idx_to_name = cal.index_to_qubit_name
            for i in indices:
                qn = idx_to_name.get(i, "")
                if qn in cal.qubits:
                    qc = cal.qubits[qn]
                    total_err += qc.single_gate_error
                    total_err += (1.0 - qc.readout_fidelity)
            idx_set = set(indices)
            for i in indices:
                for j in cal.adjacency.get(i, set()):
                    if j in idx_set and j > i:
                        total_err += (1.0 - cal.gate_fidelity(i, j))
            return 1.0 / (1.0 + total_err)
        elif strategy == "diverse":
            edges = self._count_internal_edges(indices, cal)
            ro = self._avg_readout(indices, cal)
            return ro * (1 + edges / n)
        raise ValueError(f"Unknown strategy: {strategy}")

    def _count_internal_edges(
        self, indices: list[int], cal: DeviceCalibration
    ) -> int:
        idx_set = set(indices)
        return sum(
            1 for i in indices
            for j in cal.adjacency.get(i, set())
            if j in idx_set and j > i
        )

    def _avg_readout(
        self, indices: list[int], cal: DeviceCalibration
    ) -> float:
        itn = cal.index_to_qubit_name
        fids = [
            cal.qubits[itn[i]].readout_fidelity
            for i in indices if itn.get(i, "") in cal.qubits
        ]
        return sum(fids) / len(fids) if fids else 0.0

    def _avg_gate_fidelity(
        self, indices: list[int], cal: DeviceCalibration
    ) -> float:
        idx_set = set(indices)
        fids = [
            cal.gate_fidelity(i, j)
            for i in indices
            for j in cal.adjacency.get(i, set())
            if j in idx_set and j > i and cal.gate_fidelity(i, j) > 0
        ]
        return sum(fids) / len(fids) if fids else 0.0

    def _topology_hash(
        self, indices: list[int], cal: DeviceCalibration
    ) -> str:
        """Hash of the local subgraph topology for cross-device matching."""
        idx_set = set(indices)
        sorted_phys = sorted(indices)
        phys_to_local = {p: l for l, p in enumerate(sorted_phys)}
        edges = sorted(
            (phys_to_local[i], phys_to_local[j])
            for i in sorted_phys
            for j in cal.adjacency.get(i, set())
            if j in idx_set and j > i
        )
        canonical = f"{len(indices)}q:{edges}"
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def _per_qubit_calibration(
        self, indices: list[int], cal: DeviceCalibration
    ) -> dict[str, dict[str, float]]:
        itn = cal.index_to_qubit_name
        result = {}
        for i in indices:
            qn = itn.get(i, f"Q{i}")
            if qn in cal.qubits:
                qc = cal.qubits[qn]
                result[qn] = {
                    "t1_us": qc.t1_us, "t2_us": qc.t2_us,
                    "readout_fidelity": qc.readout_fidelity,
                    "single_gate_error": qc.single_gate_error,
                }
        return result

    def _dfs_fallback(
        self, cal: DeviceCalibration,
        circuit_edges: list[tuple[int, int]], circuit_qubits: int,
    ) -> list[dict[int, int]]:
        """DFS fallback for when VF2 is unavailable. ≤8q only."""
        if circuit_qubits > 8:
            return []
        results: set[frozenset[int]] = set()
        def expand(current: frozenset[int]) -> None:
            if len(current) == circuit_qubits:
                results.add(current)
                return
            frontier = set()
            for node in current:
                for nb in cal.adjacency.get(node, set()):
                    if nb not in current and nb > min(current):
                        frontier.add(nb)
            for node in frontier:
                expand(current | {node})
        for start in range(cal.num_qubits):
            expand(frozenset({start}))
        return [
            {sorted(sg)[i]: i for i in range(len(sg))}
            for sg in results
        ]

    def summary(self, placements: list[Placement]) -> dict[str, Any]:
        """Summary statistics of placement results."""
        if not placements:
            return {"total_placements": 0}
        by_dev: dict[str, int] = {}
        for p in placements:
            by_dev[p.device_id] = by_dev.get(p.device_id, 0) + 1
        return {
            "total_placements": len(placements),
            "devices": by_dev,
            "avg_score": round(
                sum(p.score for p in placements) / len(placements), 6
            ),
            "avg_readout_fidelity": round(
                sum(p.avg_readout_fidelity for p in placements)
                / len(placements), 4
            ),
            "unique_topologies": len(set(
                p.topology_hash for p in placements
            )),
        }
