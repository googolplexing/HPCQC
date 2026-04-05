## 1.1.1 (2026-04-05)

### DSatur Packing, IQM Batch Submission, Multiprocessing Deadlock Fix (RED-RESP-PACKING-v1.0)

11 items, 97/97 validation checks on LUMI. Parquet schema: 63 columns (was 61).

**Item 1: Cache Key Fix (`sweep/twin_simulator.py`)**
- Noiseless dedup cache key now includes `obs_hash:num_qubits:topology_hash:env_name`
- Prevents silent collision when different Hamiltonians share topology
- Critical for DiagnosticTFIM coexistence with standard TFIM in same sweep

**Items 2–5: Sweep Engine Enhancements (`sweep/sweep_engine.py`, `sweep/execution_planner.py`)**
- E2/E7 Level 1 integration: TieredExecutionPlanner wired into SweepEngine
- `seed_offset` parameter in SweepExperimentConfig — enables non-overlapping seed
  ranges across SLURM jobs (ANIMLL multi-node submission)
- YAML `measurement_stats_interval` override via `dataclasses.replace()` on NoiseConfig
- VE19 amendment: 5 new assertions for interval override + preserve behavior

**Item 6: DSatur Optimal Packing (`sweep/placement_solver.py`)**
- `_pack_dsatur()` via `rx.graph_greedy_color` — provably optimal graph coloring
- Q50 4q star: 16 rounds (optimal) vs greedy 21 rounds (24% reduction)
- Default strategy changed from `"greedy"` to `"optimal"`
- Greedy remains available via `strategy: greedy` in YAML

**Item 7: IQM Batch Submission (`backends/iqm_qpu.py`)**
- `_submit_batch()` accepts `list[QuantumCircuit]` as single batch job
- Auto-chunking at VTT 200-circuit limit with correct boundary reassembly
- Test A (single batch ordering, 6/6) and Test B (multi-batch boundary, 7/7) — merge-blocking

**Item 8: Dynamic QB Mapping (`plugins/calibration_adapters/iqm_v2.py`)**
- QB32 deactivation documented — index gap at 31 (QB1→0 ... QB31→30, QB33→31 ... QB54→52)
- Defensive assertion: `len(qubit_names) == len(name_to_idx)`
- Mapping is calibration-set dependent — documented for future IQM topology changes

**Items 9–11: CZ Fidelity Columns (`data/sweep_export.py`)**
- `per_edge_cz_fidelity` confirmed populated end-to-end (HDF5 → Parquet)
- NEW: `placement_min_cz_fidelity` — `min(per_edge_cz_fidelity)`, noise-dominant edge
- NEW: `placement_avg_cz_fidelity` — `mean(per_edge_cz_fidelity)`

**Multiprocessing Deadlock Fix — Lesson #22 (`sweep/sweep_engine.py`)**
- CRITICAL: `mp.Pool` with fork copies locked C++ mutex state from parent's numpy BLAS,
  h5py, and Aer imports. Deadlock is scale-dependent — works at 8 workers, hangs at 100+.
- FIX: `_run_pool_subprocess()` serializes work items to pickle, launches fresh subprocess
  with no inherited C++ state, runs Pool there, reads results back.
- Proven on LUMI: Pool(100) → 100/100 success in 7.43s
- Trade-off: cross-battery noiseless dedup temporarily disabled (~1s extra per sweep).
  Restoration planned for v1.2.0 via two-subprocess pattern.
- All `mp.Pool` callers audited: `sweep_engine.py`, `execution_planner.py`, `eval_runner.py`

**Device Name Correction**
- `"VTT Q50 (Aalto Helmi)"` → `"Q50"` in calibration JSON
- Aalto Helmi is the 5-qubit machine, not Q50
- Matches VTT QX API, FiQCI SLURM, and CSC documentation

**New Test Files**
- `tests/test_v111_packing_batch.py` — Test A/B/C/D (19 checks)
- `tests/test_v111_validate.sh` — SLURM validation script (full node, OMP_NUM_THREADS=1)
- `tests/fork_test*.py` — multiprocessing deadlock diagnostic suite

