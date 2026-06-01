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


def _placement_sort_key(p: "Placement") -> tuple[float, list[int]]:
    """Deterministic total-order key for placement selection (F5 invariant).

    Returns a tuple sortable in ascending order, equivalent to ranking by:

      1. ``score`` descending (best score first; negated so default ascending
         sort gives the correct direction);
      2. ``physical_indices`` ascending — tie-break on physical-qubit identity.

    The same key MUST be used by both the eager (``find_all_placements``) and
    any lazy (``top_1`` / ``top_N`` score-as-iterate, planned for W1.3)
    selection paths, so they cannot disagree on ties. Required by Red's F5
    ruling in ``RED-RESP-W1-PARALLELISM-AND-OOM-ROOTCAUSE-v1.4``: the lazy and
    full-enumerate paths must select the byte-identical placement (same
    physical-qubit mapping, same order) — otherwise a different lazy traversal
    would pick a different physical qubit set → different calibration entries
    → different noise → a silently different result.

    The previous sort-by-score-alone resolved ties via Python's stable sort
    over the ``rx.vf2_mapping`` iteration order, an implicit undocumented
    ordering that a different traversal cannot reproduce.

    ``physical_indices`` is already sorted ascending by ``find_all_placements``
    (sorted at the point of dedup so the ``seen`` set keys on a canonical form),
    so the tie-break is well-defined.
    """
    return (-p.score, p.physical_indices)


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
    # Provenance of how this placement entered the resolved set (PLACEMENT union):
    # "manual" = researcher-supplied (physical_qubits); "solver" = chosen by the
    # placement solver (find_all_placements, ranked by score). Lets the VIP's
    # ranking be audited (which chains they pinned vs which the solver picked).
    source: str = "solver"


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
            List of Placement objects, sorted by ``_placement_sort_key``:
            score descending, with ties broken by ``physical_indices``
            ascending. The tie-break is the F5 invariant from
            ``RED-RESP-W1-PARALLELISM-AND-OOM-ROOTCAUSE-v1.4``: it ensures any
            lazy/bounded variant using the same key picks the byte-identical
            placement(s) as full enumerate-then-sort.
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
                        dev_qubit_idx, f"QB{dev_qubit_idx}"
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
                    source="solver",
                ))
                self._placement_counter += 1

            print(
                f"  Device {dev_id}: {len(seen)} valid placements "
                f"for {circuit_qubits}q circuit"
            )

        all_placements.sort(key=_placement_sort_key)
        if max_placements is not None:
            all_placements = all_placements[:max_placements]
        return all_placements

    def pack_rounds(
        self,
        placements: list[Placement],
        packing_seed: int = 42,
        strategy: str = "optimal",
    ) -> list[PackingRound]:
        """Pack placements into non-overlapping execution rounds.

        Args:
            placements: List of valid placements to pack.
            packing_seed: Seed for deterministic ordering (greedy only).
            strategy: "optimal" (DSatur graph coloring, provably minimal rounds)
                      or "greedy" (legacy shuffle-and-pack, more rounds).

        Each round has maximum non-overlapping placements. Enforces
        both qubit AND edge non-overlap (required for QPU correctness).
        """
        by_device: dict[str, list[Placement]] = {}
        for p in placements:
            by_device.setdefault(p.device_id, []).append(p)

        all_rounds: list[PackingRound] = []
        round_counter = 0

        for dev_id, dev_placements in by_device.items():
            if strategy == "optimal":
                dev_rounds = self._pack_dsatur(dev_placements, dev_id)
            else:
                dev_rounds = self._pack_greedy(
                    dev_placements, dev_id, packing_seed,
                )

            for round_placements, used_qubits in dev_rounds:
                all_rounds.append(PackingRound(
                    round_id=round_counter,
                    placements=round_placements,
                    total_qubits_used=len(used_qubits),
                    device_id=dev_id,
                ))
                round_counter += 1

        return all_rounds

    def _pack_dsatur(
        self,
        placements: list[Placement],
        device_id: str,
    ) -> list[tuple[list[Placement], set[int]]]:
        """Optimal packing via DSatur graph coloring.

        Builds a conflict graph (edge = shared qubit or device edge)
        and colors it with rx.graph_greedy_color() (DSatur strategy).
        Each color = one non-overlapping round.

        For Q50 4q stars: max clique = 16, DSatur finds 16 colors.
        Provably optimal.
        """
        n = len(placements)
        if n == 0:
            return []

        cal = self._devices[device_id]

        # Pre-compute qubit sets and edge sets per placement
        p_qubits = [set(p.physical_indices) for p in placements]
        p_edges = []
        for p in placements:
            edges: set[tuple[int, int]] = set()
            pq = set(p.physical_indices)
            for qi in p.physical_indices:
                for qj in cal.adjacency.get(qi, set()):
                    if qj in pq:
                        edges.add((min(qi, qj), max(qi, qj)))
            p_edges.append(edges)

        # Build conflict graph
        conflict = rx.PyGraph()
        node_indices = [conflict.add_node(i) for i in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                if p_qubits[i] & p_qubits[j] or p_edges[i] & p_edges[j]:
                    conflict.add_edge(i, j, None)

        # DSatur graph coloring — greedy_color uses saturation ordering
        coloring = rx.graph_greedy_color(conflict)
        num_colors = max(coloring.values()) + 1 if coloring else 0

        # Group placements by color → rounds
        rounds: list[tuple[list[Placement], set[int]]] = []
        for color in range(num_colors):
            round_placements = [
                placements[i] for i in range(n) if coloring[i] == color
            ]
            used = set()
            for p in round_placements:
                used |= set(p.physical_indices)
            rounds.append((round_placements, used))

        return rounds

    def _pack_greedy(
        self,
        placements: list[Placement],
        device_id: str,
        packing_seed: int,
    ) -> list[tuple[list[Placement], set[int]]]:
        """Legacy greedy packing (pre-v1.1.1 behavior)."""
        rng = random.Random(packing_seed)
        remaining = list(placements)
        rng.shuffle(remaining)
        remaining.sort(key=lambda p: p.score, reverse=True)

        cal = self._devices[device_id]
        rounds: list[tuple[list[Placement], set[int]]] = []

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
                rounds.append((round_placements, used_qubits))

            remaining = still_remaining

        return rounds

    # --- Researcher-specified placements (PLACEMENT-1) ---

    def placements_from_names(
        self,
        qubit_name_lists: list[list[str]],
        circuit_edges: list[tuple[int, int]],
        circuit_qubits: int,
        device_id: str,
        strategy: str = "max_fidelity",
    ) -> list["Placement"]:
        """Build faithful Placements from explicit qubit-name lists, bypassing
        the subgraph search (PLACEMENT-1 / researcher placement control).

        Logical qubit ``i`` maps to ``qubit_name_lists[k][i]`` -- the F5a
        placement-keyed order (mirrors backends.noise_model._resolve_selected,
        which uses supplied names "in the given logical order"). The returned
        placements are structurally identical to ``find_all_placements`` output
        (same fields, computed via the same scoring/metric helpers), so every
        downstream consumer -- device-cal noise keying, output-path naming,
        HDF5 provenance -- is unchanged.

        Fail-loud, mirroring ``_resolve_selected``'s single-placement checks
        applied per placement:
          - each list length == ``circuit_qubits``;
          - no repeated qubit within a placement;
          - every name exists in the device calibration;
          - every circuit edge maps to a real calibrated 2q device edge.
        """
        if device_id not in self._devices:
            raise ValueError(
                f"device_id {device_id!r} not registered with the solver"
            )
        cal = self._devices[device_id]
        name_to_idx = {
            name: idx for idx, name in cal.index_to_qubit_name.items()
        }

        placements: list[Placement] = []
        for k, names in enumerate(qubit_name_lists):
            if len(names) != circuit_qubits:
                raise ValueError(
                    f"physical_qubits[{k}] has {len(names)} qubit(s) but the "
                    f"circuit needs {circuit_qubits}; they must match"
                )
            if len(set(names)) != len(names):
                dupes = sorted({n for n in names if names.count(n) > 1})
                raise ValueError(
                    f"physical_qubits[{k}] repeats qubit(s): {dupes}"
                )
            missing = [n for n in names if n not in name_to_idx]
            if missing:
                raise ValueError(
                    f"physical_qubits[{k}] not in calibration {device_id!r}: "
                    f"{missing}"
                )
            logical_idx = [name_to_idx[names[i]] for i in range(circuit_qubits)]
            for (a, b) in circuit_edges:
                ia, ib = logical_idx[a], logical_idx[b]
                if ib not in cal.adjacency.get(ia, set()):
                    raise ValueError(
                        f"physical_qubits[{k}]: circuit edge ({a},{b}) maps to "
                        f"physical pair ({names[a]},{names[b]}), which is not a "
                        f"calibrated 2q gate on {device_id!r}"
                    )
            qubit_mapping = {i: names[i] for i in range(circuit_qubits)}
            phys_indices = sorted(logical_idx)
            placements.append(Placement(
                placement_id=k,
                device_id=cal.device_id,
                device_prefix=cal.device_prefix,
                qubit_mapping=qubit_mapping,
                physical_indices=phys_indices,
                score=self._score_placement(phys_indices, cal, strategy),
                internal_edges=self._count_internal_edges(phys_indices, cal),
                avg_readout_fidelity=self._avg_readout(phys_indices, cal),
                avg_gate_fidelity=self._avg_gate_fidelity(phys_indices, cal),
                topology_hash=self._topology_hash(phys_indices, cal),
                per_qubit_calibration=self._per_qubit_calibration(
                    phys_indices, cal
                ),
                source="manual",
            ))
        return placements

    def _compose_manual_solver(
        self,
        manual: list["Placement"],
        solver: list["Placement"],
        solver_top_n: int,
    ) -> tuple[list["Placement"], dict]:
        """Compose manual ∪ solver-top-N placements, deduped, manual-first.

        PLACEMENT union (Piece 1). ``manual`` are researcher-supplied placements
        (source="manual"); ``solver`` are the score-ranked solver placements
        already fetched to depth N + len(manual) (source="solver"). Returns the
        merged list and a stats dict.

        Semantics (net-N-NEW solver chains):
          - Dedup ``solver`` against ``manual`` on ``frozenset(physical_indices)``
            (set-level; Red's F5 tie-break key). A solver placement covering the
            same qubit SET as a manual one is dropped -- the manual entry wins
            (precedence), preserving the researcher's logical ordering.
          - Keep the top ``solver_top_n`` of the survivors (ranked). Because each
            manual chain collides with at most one distinct solver placement,
            D <= len(manual), so fetching N + len(manual) guarantees >= N
            survivors UNLESS the device is exhausted of distinct chains -- in
            which case fewer than N survive (S < N) and that shortfall is
            reported, never padded or silently dropped.
          - Concatenate manual-first, then solver-ranked. Re-assign placement_id
            0..M-1 over the final order (placement_id is provenance only on the
            BYO path -- identity is the qubit-name string -- so re-id is safe).
        """
        manual_sets = {frozenset(p.physical_indices) for p in manual}
        survivors, deduped = [], 0
        for p in solver:
            if frozenset(p.physical_indices) in manual_sets:
                deduped += 1
            else:
                survivors.append(p)
        kept = survivors[:solver_top_n]
        merged = list(manual) + kept
        for new_id, p in enumerate(merged):
            p.placement_id = new_id
        stats = {
            "k_manual": len(manual),
            "n_requested": solver_top_n,
            "s_solver": len(kept),
            "d_deduped": deduped,
            "fetch_depth": solver_top_n + len(manual),
            "short_of_n": max(0, solver_top_n - len(kept)),
        }
        return merged, stats

    def resolve_placements(
        self,
        circuit_edges: list[tuple[int, int]],
        circuit_qubits: int,
        device_id: str,
        strategy: str = "max_fidelity",
        max_placements: int | None = None,
        call_limit: int = 100_000,
        manual_qubit_name_lists: list[list[str]] | None = None,
        solver_top_n: int | None = None,
    ) -> list["Placement"]:
        """Single placement-resolution seam (PLACEMENT-1).

        Three modes:
          - ``manual_qubit_name_lists`` given, ``solver_top_n`` None: solver
            bypassed, exactly the researcher's placements (today's behaviour,
            byte-identical).
          - ``manual_qubit_name_lists`` None: solver self-selects via
            ``find_all_placements`` (today's behaviour, byte-identical -- forwards
            its arguments unchanged).
          - BOTH given (PLACEMENT union): the researcher's manual placements PLUS
            the solver's top ``solver_top_n`` NEW chains (deduped against the
            manual set, net-N-new). See ``_compose_manual_solver``. The solver is
            fetched to depth ``solver_top_n + len(manual)`` so N survive dedup.

        ``max_placements``/``call_limit`` apply only to the solver path
        (``placements_from_names`` returns exactly the supplied placements).
        Noise/guardrail policy (e.g. the F5a device-calibrated single-placement
        restriction) is deliberately left to the caller, since it differs per
        executor -- this seam composes placements; it does not decide whether a
        noise environment permits more than one.
        """
        if manual_qubit_name_lists and solver_top_n is not None:
            manual = self.placements_from_names(
                qubit_name_lists=manual_qubit_name_lists,
                circuit_edges=circuit_edges,
                circuit_qubits=circuit_qubits,
                device_id=device_id,
                strategy=strategy,
            )
            solver = self.find_all_placements(
                circuit_edges=circuit_edges,
                circuit_qubits=circuit_qubits,
                device_ids=[device_id],
                strategy=strategy,
                max_placements=solver_top_n + len(manual),
                call_limit=call_limit,
            )
            merged, stats = self._compose_manual_solver(manual, solver, solver_top_n)
            short = (f", {stats['short_of_n']} short of N"
                     if stats["short_of_n"] else "")
            print(
                f"  PLACEMENT union: {stats['k_manual']} manual + "
                f"{stats['s_solver']} solver (requested {stats['n_requested']}, "
                f"fetched {stats['fetch_depth']} deep, "
                f"{stats['d_deduped']} deduped against manual{short})"
            )
            return merged
        if manual_qubit_name_lists:
            return self.placements_from_names(
                qubit_name_lists=manual_qubit_name_lists,
                circuit_edges=circuit_edges,
                circuit_qubits=circuit_qubits,
                device_id=device_id,
                strategy=strategy,
            )
        return self.find_all_placements(
            circuit_edges=circuit_edges,
            circuit_qubits=circuit_qubits,
            device_ids=[device_id],
            strategy=strategy,
            max_placements=max_placements,
            call_limit=call_limit,
        )

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
        """Hash of the abstract graph isomorphism class.

        Uses degree sequence + edge count as the canonical form.
        This correctly groups all placements with the same abstract
        topology regardless of physical qubit labeling:
          P4 (path):   degrees [1,1,2,2], 3 edges
          K1,3 (star): degrees [1,1,1,3], 3 edges
          C4 (square): degrees [2,2,2,2], 4 edges

        For circuits ≤8 qubits, degree sequence + edge count is a
        complete graph invariant (no two non-isomorphic connected
        graphs share both). For larger circuits, switch to nauty-based
        canonical labeling if needed (v1.2.0).

        RED-RESP-TOPOLOGY-DIVERSITY-v1.0 §3.2, Approach A.
        """
        idx_set = set(indices)
        n = len(indices)

        # Build local edge list
        edges = []
        for i in indices:
            for j in cal.adjacency.get(i, set()):
                if j in idx_set and j > i:
                    edges.append((i, j))

        # Compute degree of each vertex in the local subgraph
        degree_count = {i: 0 for i in indices}
        for i, j in edges:
            degree_count[i] += 1
            degree_count[j] += 1

        # Canonical form: sorted degree sequence + edge count
        degrees = sorted(degree_count.values())
        canonical = f"{n}q:deg{degrees}:e{len(edges)}"
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
