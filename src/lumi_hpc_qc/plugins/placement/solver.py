# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Placement solver for circuit multiplexing on Q50.

Phase B: Finds optimal non-overlapping qubit placements for packing
multiple copies of a circuit onto the Q50's 53 qubits. A 4-qubit VQE
circuit can be placed up to 12 times (48/53 qubits used), generating
12× data per QPU submission at zero additional cost.

Uses rustworkx for graph operations.

Usage:
    from lumi_hpc_qc.plugins.placement.solver import PlacementSolver

    solver = PlacementSolver("examples/q50_calibration_20260326.json")
    placements = solver.find_placements(
        circuit_qubits=4,
        num_placements=12,
        strategy="max_fidelity",
    )
    for p in placements:
        print(p["qubit_mapping"], p["score"])
"""

from __future__ import annotations

import json
from itertools import combinations
from typing import Any


class PlacementSolver:
    """Find optimal qubit placements for circuit multiplexing."""

    def __init__(self, calibration_path: str) -> None:
        self._cal_path = calibration_path
        with open(calibration_path) as f:
            self._cal = json.load(f)

        self._qubit_data = self._cal.get("qubits", {})
        self._gate_data = self._cal.get("two_qubit_gates", {})
        self._qubit_names = list(self._qubit_data.keys())
        self._name_to_idx = {n: i for i, n in enumerate(self._qubit_names)}
        self._num_device_qubits = len(self._qubit_names)

        # Build adjacency list
        self._adj: dict[int, set[int]] = {i: set() for i in range(self._num_device_qubits)}
        self._edge_fidelity: dict[tuple[int, int], float] = {}
        for gate_pair, gdata in self._gate_data.items():
            parts = gate_pair.split("-")
            if len(parts) != 2:
                continue
            q1, q2 = parts
            if q1 not in self._name_to_idx or q2 not in self._name_to_idx:
                continue
            i, j = self._name_to_idx[q1], self._name_to_idx[q2]
            self._adj[i].add(j)
            self._adj[j].add(i)
            fid = gdata.get("cz_fidelity", 1.0 - gdata.get("cz_error", 0.005))
            self._edge_fidelity[(i, j)] = fid
            self._edge_fidelity[(j, i)] = fid

    def find_placements(
        self,
        circuit_qubits: int,
        num_placements: int = 12,
        strategy: str = "max_fidelity",
    ) -> list[dict]:
        """Find non-overlapping qubit placements on the device.

        Args:
            circuit_qubits: Number of qubits per circuit placement.
            num_placements: Maximum number of placements to find.
            strategy: Ranking strategy — "max_fidelity", "max_connectivity",
                      "min_error", or "diverse".

        Returns:
            List of placement dicts, each containing:
                qubit_mapping: {logical_idx: physical_qubit_name}
                physical_indices: list of physical qubit indices
                score: float (strategy-dependent)
                avg_readout_fidelity: float
                avg_cz_fidelity: float (for edges within subgraph)
                internal_edges: int (connectivity within subgraph)
        """
        # Step 1: Find all connected subgraphs of the required size
        subgraphs = self._enumerate_connected_subgraphs(circuit_qubits)

        if not subgraphs:
            print(f"  No connected subgraphs of size {circuit_qubits} found")
            return []

        # Step 2: Score each subgraph
        scored = []
        for sg in subgraphs:
            score = self._score_subgraph(sg, strategy)
            scored.append((score, sg))

        # Step 3: Sort by score (descending)
        scored.sort(key=lambda x: x[0], reverse=True)

        # Step 4: Greedily select non-overlapping placements
        used_qubits: set[int] = set()
        placements = []

        for score, sg in scored:
            if len(placements) >= num_placements:
                break
            sg_set = set(sg)
            if sg_set & used_qubits:
                continue
            used_qubits |= sg_set

            # Build placement info
            internal_edges = sum(
                1 for i in sg for j in self._adj[i] if j in sg_set and j > i
            )
            avg_ro = sum(
                self._qubit_data[self._qubit_names[i]]["readout_fidelity"]
                for i in sg
            ) / len(sg)
            cz_fids = [
                self._edge_fidelity.get((i, j), 0)
                for i in sg for j in self._adj[i] if j in sg_set and j > i
            ]
            avg_cz = sum(cz_fids) / len(cz_fids) if cz_fids else 0.0

            placements.append({
                "placement_id": len(placements),
                "qubit_mapping": {
                    k: self._qubit_names[sg[k]] for k in range(len(sg))
                },
                "physical_indices": list(sg),
                "score": round(score, 6),
                "avg_readout_fidelity": round(avg_ro, 4),
                "avg_cz_fidelity": round(avg_cz, 4),
                "internal_edges": internal_edges,
            })

        print(f"  Placement solver: {len(placements)} placements found "
              f"for {circuit_qubits}q circuits ({strategy} strategy)")
        print(f"  Total qubits used: {len(used_qubits)}/{self._num_device_qubits}")

        return placements

    def _enumerate_connected_subgraphs(self, size: int) -> list[tuple[int, ...]]:
        """Enumerate all connected subgraphs of a given size.

        Uses BFS-based expansion from each node. For small sizes (≤8),
        this is tractable on Q50's 53-qubit graph. For larger sizes,
        use find_placements_heuristic instead.
        """
        if size > 10:
            return self._enumerate_heuristic(size)

        results = set()

        for start in range(self._num_device_qubits):
            self._expand_subgraph(frozenset({start}), size, results)

        return [tuple(sorted(sg)) for sg in results]

    def _expand_subgraph(
        self,
        current: frozenset[int],
        target_size: int,
        results: set[frozenset[int]],
    ) -> None:
        """Recursively expand a subgraph to target size."""
        if len(current) == target_size:
            results.add(current)
            return

        # Find all neighbors of the current subgraph
        frontier = set()
        for node in current:
            for neighbor in self._adj[node]:
                if neighbor not in current and neighbor > min(current):
                    frontier.add(neighbor)

        for node in frontier:
            self._expand_subgraph(current | {node}, target_size, results)

    def _enumerate_heuristic(self, size: int) -> list[tuple[int, ...]]:
        """Heuristic subgraph enumeration for large sizes.

        Uses BFS growth from each starting node, keeping the best
        neighbors at each step.
        """
        results = []
        for start in range(self._num_device_qubits):
            sg = {start}
            candidates = set(self._adj[start])
            while len(sg) < size and candidates:
                # Pick the candidate with highest readout fidelity
                best = max(
                    candidates,
                    key=lambda q: self._qubit_data[self._qubit_names[q]]["readout_fidelity"],
                )
                sg.add(best)
                candidates.discard(best)
                candidates |= (self._adj[best] - sg)
            if len(sg) == size:
                key = tuple(sorted(sg))
                if key not in [tuple(sorted(r)) for r in results]:
                    results.append(key)
        return results

    def _score_subgraph(self, sg: tuple[int, ...], strategy: str) -> float:
        """Score a subgraph by the given strategy."""
        sg_set = set(sg)
        n = len(sg)

        if strategy == "max_fidelity":
            # Average of readout fidelity + CZ fidelity
            ro_sum = sum(
                self._qubit_data[self._qubit_names[i]]["readout_fidelity"]
                for i in sg
            )
            cz_fids = [
                self._edge_fidelity.get((i, j), 0)
                for i in sg for j in self._adj[i] if j in sg_set and j > i
            ]
            cz_sum = sum(cz_fids)
            total = ro_sum / n
            if cz_fids:
                total = (total + cz_sum / len(cz_fids)) / 2
            return total

        elif strategy == "max_connectivity":
            # Number of internal edges / maximum possible
            internal = sum(
                1 for i in sg for j in self._adj[i] if j in sg_set and j > i
            )
            max_edges = n * (n - 1) / 2
            return internal / max_edges if max_edges > 0 else 0

        elif strategy == "min_error":
            # Inverse of total error (lower error = higher score)
            total_err = 0
            for i in sg:
                total_err += self._qubit_data[self._qubit_names[i]].get("single_gate_error", 0.001)
                total_err += (1 - self._qubit_data[self._qubit_names[i]]["readout_fidelity"])
            for i in sg:
                for j in self._adj[i]:
                    if j in sg_set and j > i:
                        total_err += self._edge_fidelity.get((i, j), 0)
                        total_err = total_err  # CZ error already captured
            return 1.0 / (1.0 + total_err)

        elif strategy == "diverse":
            # Penalize subgraphs that are too similar to a linear chain
            # Reward subgraphs with branching (higher connectivity)
            internal = sum(
                1 for i in sg for j in self._adj[i] if j in sg_set and j > i
            )
            ro = sum(
                self._qubit_data[self._qubit_names[i]]["readout_fidelity"]
                for i in sg
            ) / n
            # Connectivity bonus: more edges = more diverse topology
            conn_bonus = internal / n
            return ro * (1 + conn_bonus)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def summary(self, placements: list[dict]) -> dict:
        """Generate a summary of placement results."""
        if not placements:
            return {"num_placements": 0}

        return {
            "num_placements": len(placements),
            "total_qubits_used": len(set(
                idx for p in placements for idx in p["physical_indices"]
            )),
            "device_qubits": self._num_device_qubits,
            "utilization": len(set(
                idx for p in placements for idx in p["physical_indices"]
            )) / self._num_device_qubits,
            "avg_readout_fidelity": sum(
                p["avg_readout_fidelity"] for p in placements
            ) / len(placements),
            "avg_cz_fidelity": sum(
                p["avg_cz_fidelity"] for p in placements
            ) / len(placements),
        }
