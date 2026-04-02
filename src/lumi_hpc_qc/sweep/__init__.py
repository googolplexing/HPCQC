# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E sweep engine — systematic noise atlas generation.

This package orchestrates the complete sweep pipeline:
  - placement_solver: VF2 subgraph isomorphism, multi-device placement
  - hdf5_writer: HDF5-first write-during-execution (data layer, in data/)
  - (future) engine: top-level sweep orchestrator
  - (future) twin_simulator: multi-calibration Aer battery
  - (future) execution_planner: tiered CPU/GPU backend selection
  - (future) circuit_composer: mixed-experiment QPU circuit composition
  - (future) demultiplexer: bitstring extraction (from Phase C)
  - (future) byo_ingestion: QPY/QASM/script circuit loading
"""
