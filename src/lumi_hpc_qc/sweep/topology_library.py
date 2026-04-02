# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Topology library — reference circuit topologies for sweep engine.

Defines the set of abstract graph topologies that the sweep engine
enumerates against each device's coupling map. For each topology,
VF2 finds all valid physical qubit placements. The sweep then runs
a reference circuit (BYO, evaluation-only) on each placement.

This library is extensible — researchers add entries as Python dicts.
No framework code changes needed.

RED-RESP-TOPOLOGY-DIVERSITY-v1.0 §4
"""

from __future__ import annotations

from typing import Any

# Registry of reference topologies.
# Each entry: {qubits: int, edges: [(i, j), ...], description: str}
# The solver runs VF2 for each entry against each device coupling map.
# Entries with 0 valid placements on a device are silently skipped.

TOPOLOGY_LIBRARY: dict[str, dict[str, Any]] = {
    # 2-qubit — only one connected graph
    "2q_pair": {
        "qubits": 2,
        "edges": [(0, 1)],
        "description": "Single edge (trivial 2-qubit coupling)",
    },

    # 4-qubit — three distinct classes on square lattice
    "4q_chain": {
        "qubits": 4,
        "edges": [(0, 1), (1, 2), (2, 3)],
        "description": "Path graph P4 — linear chain",
    },
    "4q_star": {
        "qubits": 4,
        "edges": [(0, 1), (0, 2), (0, 3)],
        "description": "Star graph K1,3 — hub connected to 3 leaves",
    },
    "4q_square": {
        "qubits": 4,
        "edges": [(0, 1), (1, 2), (2, 3), (3, 0)],
        "description": "Cycle graph C4 — 4-qubit ring",
    },

    # 6-qubit
    "6q_chain": {
        "qubits": 6,
        "edges": [(i, i + 1) for i in range(5)],
        "description": "Path graph P6 — 6-qubit linear chain",
    },
    "6q_star": {
        "qubits": 6,
        "edges": [(0, i) for i in range(1, 6)],
        "description": "Star graph K1,5 — hub connected to 5 leaves",
    },

    # 8-qubit
    "8q_chain": {
        "qubits": 8,
        "edges": [(i, i + 1) for i in range(7)],
        "description": "Path graph P8 — 8-qubit linear chain",
    },
}


def get_topologies_for_size(num_qubits: int) -> dict[str, dict[str, Any]]:
    """Return all library entries matching a qubit count."""
    return {
        name: spec
        for name, spec in TOPOLOGY_LIBRARY.items()
        if spec["qubits"] == num_qubits
    }


def list_all_topologies() -> list[str]:
    """Return sorted list of all topology names in the library."""
    return sorted(TOPOLOGY_LIBRARY.keys())
