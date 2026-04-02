# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E sweep engine — systematic noise atlas generation.

This package orchestrates the complete sweep pipeline:
  - placement_solver: VF2 subgraph isomorphism, multi-device placement
  - topology_library: reference circuit topologies for multi-topology sweeps
  - execution_planner: tiered CPU/GPU backend selection + parallel dispatch
  - circuit_loader: BYO circuit ingestion (QPY/QASM/script)
  - eval_runner: evaluation-only mode for fixed circuits
  - hdf5_writer: HDF5-first write-during-execution (data layer, in data/)
  - (future) twin_simulator: multi-calibration Aer battery (E4)
  - (future) engine: top-level sweep orchestrator (E7)
  - (future) circuit_composer: mixed-experiment QPU circuit composition (E6b)
  - (future) demultiplexer: bitstring extraction (from Phase C)
"""
