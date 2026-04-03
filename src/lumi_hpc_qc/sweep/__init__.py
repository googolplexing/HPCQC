# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E sweep engine — systematic noise atlas generation.

This package orchestrates the complete sweep pipeline:
  - placement_solver: VF2 subgraph isomorphism, multi-device placement (E1)
  - topology_library: reference circuit topologies for multi-topology sweeps
  - execution_planner: tiered CPU/GPU backend selection + parallel dispatch (E2)
  - circuit_loader: BYO circuit ingestion (QPY/QASM/script) (E5)
  - eval_runner: evaluation-only mode for fixed circuits (E5)
  - hdf5_writer: HDF5-first write-during-execution (data layer, in data/) (E3)
  - noise_configs: 11 noise environment definitions (E4)
  - twin_simulator: multi-calibration Aer battery (E4)
  - circuit_composer: compose placements into device-width circuits (E6a)
  - demultiplexer: extract per-placement counts from composites (E6a)
  - round_executor: multi-round packed circuit orchestrator (E6a)
  - sweep_engine: top-level orchestrator — YAML to HDF5 noise atlas (E7)
  - (future) sweep_export: HDF5 → Parquet 61-column export (E8)
  - (future) circuit_composer mixed-experiment packing (E6b)
"""
