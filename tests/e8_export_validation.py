#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E — E8: Sweep Export (HDF5 → 61-Column Parquet) Validation.

Tests the export pipeline that converts E7's HDF5 noise atlas into
the 61-column Parquet training table defined in RED-DIRECTIVE-E4-SCHEMA-v1.0 §4.
This Parquet file is the Team Orange data interface.

Validation targets:
  VE16: No raw histograms in Parquet (aggregated features only)
  VE17: Calibration columns present and populated
  VE23: Topology columns populated from topology library metadata

Additional checks:
  - Schema has exactly 61 columns
  - All column types match the spec
  - Every HDF5 leaf group becomes exactly one Parquet row
  - Per-qubit calibration arrays have correct length
  - Noise environment metadata correctly enriched
  - Summary CSV written alongside Parquet
  - Round-trip: Parquet values match HDF5 source data

Run on LUMI standard partition (CPU only):
    srun ... python tests/e8_export_validation.py

Expected: E8 VALIDATION: ALL CHECKS PASSED

RED-SPEC-002 §9
RED-DIRECTIVE-E4-SCHEMA-v1.0 §4–§5
"""

import sys
import os
import json
import time
import tempfile
import traceback

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

import numpy as np
import h5py

passed = 0
failed = 0
errors = []


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  [PASS] {name}")
        passed += 1
    else:
        print(f"  [FAIL] {name}: {detail}")
        failed += 1
        errors.append(f"{name}: {detail}")


# ═══════════════════════════════════════════════════════════════════════
# Step 0: Generate a small HDF5 via E7 for export testing
# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.0: Generate Test HDF5 via E7 Sweep Engine ===")
try:
    from lumi_hpc_qc.sweep.sweep_engine import run_sweep_from_dict

    cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
    if not os.path.exists(cal_path):
        cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260326.json")
    assert os.path.exists(cal_path), f"No calibration file found"

    test_dir = tempfile.mkdtemp(prefix="e8_test_")

    # Small sweep: chain + star, 1 seed, 2 noise configs
    sweep_yaml = {
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "topologies": ["4q_chain", "4q_star"],
                "seeds": 1,
                "noise_configs": ["noiseless", "noise_full"],
                "placement": "all_valid",
            }],
            "calibrations": [cal_path],
            "execution": {"cpu_workers": 4},
            "output_dir": test_dir,
            "hdf5_filename": "test_sweep.h5",
            "sweep_id": "e8_export_test",
        }
    }

    print("  Running small sweep for export testing...")
    t0 = time.time()
    sweep_result = run_sweep_from_dict(sweep_yaml, device="CPU")
    t_sweep = time.time() - t0
    print(f"  Sweep completed in {t_sweep:.1f}s: {sweep_result.total_hdf5_writes} writes")

    hdf5_path = sweep_result.hdf5_path
    check("E8.0: Test HDF5 generated",
          os.path.exists(hdf5_path) and sweep_result.total_hdf5_writes > 0,
          f"writes={sweep_result.total_hdf5_writes}, errors={sweep_result.total_errors}")

    # Count leaf groups for later comparison
    h5 = h5py.File(hdf5_path, "r")
    hdf5_leaf_count_list = [0]
    def _count(name, obj):
        if isinstance(obj, h5py.Group) and "energy_trajectory" in obj:
            hdf5_leaf_count_list[0] += 1
    h5.visititems(_count)
    h5.close()
    hdf5_leaf_count = hdf5_leaf_count_list[0]
    print(f"  HDF5 leaf groups: {hdf5_leaf_count}")

except Exception as e:
    check("E8.0 test HDF5 generation", False, traceback.format_exc())
    # Can't continue without test data
    print(f"\nE8 VALIDATION: CANNOT CONTINUE (no test data)")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.1: Schema Definition ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.data.sweep_export import _build_parquet_schema

    schema = _build_parquet_schema()

    check("Schema has 61 columns", len(schema) == 61,
          f"got {len(schema)}")

    # Verify column names match the spec
    expected_columns = [
        # Identity & Provenance (4)
        "experiment_id", "schema_version", "framework_version", "quality_gate_passed",
        # Experiment Configuration (11)
        "model", "ansatz", "optimizer", "gradient_method", "initializer",
        "num_qubits", "ansatz_reps", "num_parameters", "optimizer_maxiter",
        "shots", "seed",
        # Hamiltonian Properties (3)
        "spectral_gap", "hamiltonian_locality", "num_pauli_terms",
        # Device & Placement (7)
        "device", "placement_qubits", "circuit_topology",
        "topology_equivalence_class", "placement_fidelity_score",
        "submission_round", "coupling_map_source",
        # Calibration (8)
        "calibration_source", "calibration_device", "calibration_date",
        "calibration_is_synthetic",
        "per_qubit_t1_us", "per_qubit_t2_us", "per_qubit_readout_fidelity",
        "per_edge_cz_fidelity",
        # Noise & Mitigation (4)
        "noise_environment", "noise_channels_active",
        "mitigation_readout", "mitigation_zne",
        # Circuit Metrics (3)
        "pre_transpilation_depth", "post_transpilation_depth", "swap_count",
        # Results (7)
        "best_energy", "exact_energy", "relative_error",
        "total_iterations", "optimizer_converged", "wall_time_s", "noiseless_tier",
        # Aggregated Features (3)
        "convergence_rate", "energy_variance", "final_gradient_norm",
        # Noise Fingerprinting (11)
        "measurement_entropy", "dominant_bitstring_fraction", "num_unique_bitstrings",
        "bitstring_hamming_weight_mean", "bitstring_hamming_weight_variance",
        "z_group_expectation_mean", "xz_expectation_ratio",
        "effective_hilbert_dimension", "kl_divergence_from_uniform",
        "expectation_variance_across_groups", "dominant_bitstring_hamming_weight",
    ]

    actual_columns = [f.name for f in schema]
    check("All 61 expected columns present",
          actual_columns == expected_columns,
          f"missing: {set(expected_columns) - set(actual_columns)}, "
          f"extra: {set(actual_columns) - set(expected_columns)}")

    # Verify key column types
    import pyarrow as pa
    type_checks = {
        "experiment_id": pa.string(),
        "quality_gate_passed": pa.bool_(),
        "num_qubits": pa.int32(),
        "best_energy": pa.float64(),
        "seed": pa.int32(),
        "per_qubit_t1_us": pa.list_(pa.float64()),
        "placement_fidelity_score": pa.float64(),
        "optimizer_converged": pa.bool_(),
    }
    for col_name, expected_type in type_checks.items():
        field = schema.field(col_name)
        check(f"Type: {col_name} is {expected_type}",
              field.type == expected_type,
              f"got {field.type}")

except Exception as e:
    check("E8.1 schema definition", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.2: Export Execution ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.data.sweep_export import export_sweep_to_parquet

    parquet_path = os.path.join(test_dir, "test_sweep.parquet")

    export_result = export_sweep_to_parquet(
        hdf5_path,
        parquet_path,
        include_csv=True,
    )

    check("Export completed", export_result["total_rows"] > 0,
          f"rows={export_result['total_rows']}")

    check("Parquet file created", os.path.exists(parquet_path))

    check("Row count matches HDF5 leaf count",
          export_result["total_rows"] == hdf5_leaf_count,
          f"parquet={export_result['total_rows']}, hdf5={hdf5_leaf_count}")

    check("61 columns in export", export_result["columns"] == 61,
          f"got {export_result['columns']}")

    csv_path = export_result.get("csv_path")
    check("Summary CSV created",
          csv_path is not None and os.path.exists(csv_path))

except Exception as e:
    check("E8.2 export execution", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.3: Parquet Content Verification ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    import pyarrow.parquet as pq

    table = pq.read_table(parquet_path)
    df_cols = table.column_names

    check("Parquet readable", table is not None)
    check("Parquet has 61 columns", len(df_cols) == 61,
          f"got {len(df_cols)}")
    check("Parquet row count matches",
          table.num_rows == hdf5_leaf_count,
          f"parquet={table.num_rows}, hdf5={hdf5_leaf_count}")

    # ── Check non-null columns that should always be populated ──
    for col in ["experiment_id", "model", "device", "noise_environment",
                "seed", "num_qubits", "best_energy", "framework_version"]:
        col_data = table.column(col)
        null_count = col_data.null_count
        check(f"Column '{col}' has no nulls",
              null_count == 0,
              f"{null_count} nulls in {table.num_rows} rows")

    # ── Check model is "tfim" for all rows ──
    models = table.column("model").to_pylist()
    check("All rows have model='tfim'",
          all(m == "tfim" for m in models),
          f"unique models: {set(models)}")

    # ── Check both topologies present ──
    topologies = set(table.column("circuit_topology").to_pylist())
    check("Chain topology present", "4q_chain" in topologies,
          f"topologies: {topologies}")
    check("Star topology present", "4q_star" in topologies,
          f"topologies: {topologies}")

    # ── Check both noise environments ──
    envs = set(table.column("noise_environment").to_pylist())
    check("Noiseless environment in Parquet", "noiseless" in envs,
          f"envs: {envs}")
    check("noise_full environment in Parquet", "noise_full" in envs,
          f"envs: {envs}")

    # ── Check seeds ──
    seeds = set(table.column("seed").to_pylist())
    check("Seed 0 in Parquet", 0 in seeds)

    # ── Check num_qubits ──
    nq_vals = set(table.column("num_qubits").to_pylist())
    check("num_qubits = 4 for all rows", nq_vals == {4},
          f"got {nq_vals}")

except Exception as e:
    check("E8.3 parquet content", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.4: VE16 — No Raw Histograms in Parquet ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # VE16: The Parquet must NOT contain raw measurement histograms.
    # Only aggregated features (measurement_entropy, dominant_bitstring_fraction, etc.)
    # are allowed. Raw counts stay in HDF5 only.

    col_names_set = set(df_cols)

    # These columns should NOT exist
    forbidden = [
        "counts", "raw_counts", "measurement_stats", "bitstring_counts",
        "histogram", "raw_histogram",
    ]
    for forbidden_col in forbidden:
        check(f"VE16: no '{forbidden_col}' column in Parquet",
              forbidden_col not in col_names_set)

    # Verify aggregated features ARE present (the allowed form)
    for agg_col in ["measurement_entropy", "dominant_bitstring_fraction",
                     "num_unique_bitstrings"]:
        check(f"VE16: aggregated '{agg_col}' column exists",
              agg_col in col_names_set)

    # Verify no column contains JSON strings of count dicts
    # (paranoia check — someone might sneak raw counts as a string column)
    for col_name in df_cols:
        col_data = table.column(col_name)
        if col_data.type == pa.string():
            # Sample first non-null value
            for val in col_data.to_pylist()[:10]:
                if val and isinstance(val, str) and val.startswith("{"):
                    try:
                        parsed = json.loads(val)
                        if isinstance(parsed, dict) and all(isinstance(v, int) for v in parsed.values()):
                            check(f"VE16: column '{col_name}' does not contain count dicts",
                                  False, f"found dict with int values: {list(parsed.keys())[:3]}...")
                            break
                    except json.JSONDecodeError:
                        pass

except Exception as e:
    check("VE16 no histograms", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.5: VE17 — Calibration Columns ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # VE17: Calibration columns must be present and populated

    check("VE17: calibration_source column exists",
          "calibration_source" in col_names_set)
    check("VE17: calibration_device column exists",
          "calibration_device" in col_names_set)

    # Check calibration source is populated
    cal_sources = table.column("calibration_source").to_pylist()
    check("VE17: calibration_source populated",
          all(s and len(s) > 0 for s in cal_sources),
          f"sample: {cal_sources[:3]}")

    # Check per-qubit calibration arrays
    t1_col = table.column("per_qubit_t1_us").to_pylist()
    t2_col = table.column("per_qubit_t2_us").to_pylist()
    ro_col = table.column("per_qubit_readout_fidelity").to_pylist()

    # For noisy environments, calibration arrays should be populated
    noise_envs_list = table.column("noise_environment").to_pylist()
    noisy_indices = [i for i, e in enumerate(noise_envs_list) if e != "noiseless"]

    if noisy_indices:
        sample_idx = noisy_indices[0]
        t1_sample = t1_col[sample_idx]
        t2_sample = t2_col[sample_idx]
        ro_sample = ro_col[sample_idx]

        check("VE17: per_qubit_t1_us has 4 values (4q circuit)",
              t1_sample is not None and len(t1_sample) == 4,
              f"got {t1_sample}")

        check("VE17: per_qubit_t2_us has 4 values",
              t2_sample is not None and len(t2_sample) == 4,
              f"got {t2_sample}")

        check("VE17: per_qubit_readout_fidelity has 4 values",
              ro_sample is not None and len(ro_sample) == 4,
              f"got {ro_sample}")

        # Sanity: T1 values should be positive (microseconds)
        if t1_sample:
            check("VE17: T1 values are positive",
                  all(v > 0 for v in t1_sample),
                  f"t1 values: {t1_sample}")

        # Sanity: readout fidelity should be between 0 and 1
        if ro_sample:
            check("VE17: readout fidelity in [0, 1]",
                  all(0 <= v <= 1 for v in ro_sample),
                  f"readout values: {ro_sample}")

except Exception as e:
    check("VE17 calibration columns", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.6: VE23 — Topology Columns ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # VE23: Topology-related columns must be populated

    topo_col = table.column("circuit_topology").to_pylist()
    eq_col = table.column("topology_equivalence_class").to_pylist()
    score_col = table.column("placement_fidelity_score").to_pylist()

    check("VE23: circuit_topology populated",
          all(t and len(t) > 0 for t in topo_col),
          f"sample: {topo_col[:3]}")

    check("VE23: topology_equivalence_class populated",
          all(t and len(t) > 0 for t in eq_col),
          f"sample: {eq_col[:3]}")

    check("VE23: placement_fidelity_score populated",
          all(isinstance(s, float) and s > 0 for s in score_col),
          f"sample: {score_col[:3]}")

    # Different topologies should have different equivalence classes
    chain_hashes = set(eq_col[i] for i in range(len(topo_col)) if topo_col[i] == "4q_chain")
    star_hashes = set(eq_col[i] for i in range(len(topo_col)) if topo_col[i] == "4q_star")
    check("VE23: chain and star have different topology hashes",
          chain_hashes != star_hashes,
          f"chain: {chain_hashes}, star: {star_hashes}")

except Exception as e:
    check("VE23 topology columns", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.7: Noise Environment Metadata ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    channels_col = table.column("noise_channels_active").to_pylist()
    tier_col = table.column("noiseless_tier").to_pylist()
    shots_col = table.column("shots").to_pylist()

    # Noiseless rows should have channels="none", tier=0, shots=0
    for i, env in enumerate(noise_envs_list):
        if env == "noiseless":
            check("Noise meta: noiseless has channels='none'",
                  channels_col[i] == "none")
            check("Noise meta: noiseless has tier=0",
                  tier_col[i] == 0)
            check("Noise meta: noiseless has shots=0",
                  shots_col[i] == 0)
            break

    # noise_full rows should have all channels active
    for i, env in enumerate(noise_envs_list):
        if env == "noise_full":
            check("Noise meta: noise_full has non-empty channels",
                  channels_col[i] is not None and len(channels_col[i]) > 10,
                  f"got: {channels_col[i]}")
            check("Noise meta: noise_full has tier=3",
                  tier_col[i] == 3)
            check("Noise meta: noise_full has shots=4096",
                  shots_col[i] == 4096)
            break

except Exception as e:
    check("E8.7 noise metadata", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.8: Round-Trip Verification ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Verify key values in Parquet match the HDF5 source
    h5 = h5py.File(hdf5_path, "r")

    # Pick a leaf group, find it in Parquet
    sample_holder = [None, None]  # [path, grp]
    def _find_sample(name, obj):
        if sample_holder[0] is None and isinstance(obj, h5py.Group) and "energy_trajectory" in obj:
            sample_holder[0] = name
            sample_holder[1] = obj
    h5.visititems(_find_sample)

    sample_path = sample_holder[0]
    sample_grp = sample_holder[1]

    if sample_grp is not None:
        h5_experiment_id = str(sample_grp.attrs.get("experiment_id", ""))
        h5_best_energy = float(sample_grp.attrs.get("best_energy", 0.0))
        h5_seed = int(sample_grp.attrs.get("seed", -1))
        h5_noise = str(sample_grp.attrs.get("noise_config", ""))

        # Find matching row in Parquet
        exp_ids = table.column("experiment_id").to_pylist()
        if h5_experiment_id in exp_ids:
            pq_idx = exp_ids.index(h5_experiment_id)
            pq_energy = table.column("best_energy").to_pylist()[pq_idx]
            pq_seed = table.column("seed").to_pylist()[pq_idx]
            pq_noise = table.column("noise_environment").to_pylist()[pq_idx]

            check("Round-trip: best_energy matches",
                  abs(pq_energy - h5_best_energy) < 1e-12,
                  f"h5={h5_best_energy}, pq={pq_energy}")
            check("Round-trip: seed matches",
                  pq_seed == h5_seed)
            check("Round-trip: noise_environment matches",
                  pq_noise == h5_noise)
        else:
            check("Round-trip: experiment_id found in Parquet",
                  False, f"'{h5_experiment_id}' not in Parquet")

    h5.close()

except Exception as e:
    check("E8.8 round-trip", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.9: Derived Features ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.data.sweep_export import (
        _convergence_rate, _energy_variance,
        _measurement_entropy, _dominant_bitstring_fraction,
        _effective_hilbert_dimension, _kl_divergence_from_uniform,
    )

    # Test convergence_rate with known trajectory
    traj_converging = [10.0, 8.0, 6.0, 5.0, 4.5, 4.3, 4.2, 4.15, 4.12, 4.11]
    cr = _convergence_rate(traj_converging)
    check("Derived: convergence_rate is negative (improving)",
          cr is not None and cr < 0,
          f"got {cr}")

    traj_flat = [4.0] * 10
    cr_flat = _convergence_rate(traj_flat)
    check("Derived: flat trajectory has ~0 convergence_rate",
          cr_flat is not None and abs(cr_flat) < 1e-10,
          f"got {cr_flat}")

    # Test energy_variance
    ev = _energy_variance(traj_converging)
    check("Derived: energy_variance is non-negative",
          ev is not None and ev >= 0)

    ev_flat = _energy_variance(traj_flat)
    check("Derived: flat trajectory has 0 variance",
          ev_flat is not None and abs(ev_flat) < 1e-10)

    # Test measurement_entropy
    uniform_counts = {"00": 256, "01": 256, "10": 256, "11": 256}
    ent = _measurement_entropy(uniform_counts)
    check("Derived: uniform counts have entropy=2.0 (2 qubits)",
          ent is not None and abs(ent - 2.0) < 1e-10,
          f"got {ent}")

    peaked_counts = {"00": 1024}
    ent_peaked = _measurement_entropy(peaked_counts)
    check("Derived: single-bitstring has entropy=0",
          ent_peaked is not None and abs(ent_peaked) < 1e-10,
          f"got {ent_peaked}")

    # Test dominant_bitstring_fraction
    dbf = _dominant_bitstring_fraction(uniform_counts)
    check("Derived: uniform has dbf=0.25",
          dbf is not None and abs(dbf - 0.25) < 1e-10)

    dbf_peaked = _dominant_bitstring_fraction(peaked_counts)
    check("Derived: peaked has dbf=1.0",
          dbf_peaked is not None and abs(dbf_peaked - 1.0) < 1e-10)

    # Test effective_hilbert_dimension (participation ratio)
    ehd_uniform = _effective_hilbert_dimension(uniform_counts)
    check("Derived: uniform has ehd=4.0 (2^2 states)",
          ehd_uniform is not None and abs(ehd_uniform - 4.0) < 1e-10,
          f"got {ehd_uniform}")

    ehd_peaked = _effective_hilbert_dimension(peaked_counts)
    check("Derived: peaked has ehd=1.0",
          ehd_peaked is not None and abs(ehd_peaked - 1.0) < 1e-10,
          f"got {ehd_peaked}")

    # Test KL divergence
    kl_uniform = _kl_divergence_from_uniform(uniform_counts, 2)
    check("Derived: uniform has KL=0",
          kl_uniform is not None and abs(kl_uniform) < 1e-10,
          f"got {kl_uniform}")

    kl_peaked = _kl_divergence_from_uniform(peaked_counts, 2)
    check("Derived: peaked has KL=2.0 (log2(4))",
          kl_peaked is not None and abs(kl_peaked - 2.0) < 1e-10,
          f"got {kl_peaked}")

    # Test null handling
    check("Derived: None counts → None entropy",
          _measurement_entropy(None) is None)
    check("Derived: None counts → None dbf",
          _dominant_bitstring_fraction(None) is None)
    check("Derived: short trajectory → None convergence_rate",
          _convergence_rate([1.0, 2.0]) is None)

except Exception as e:
    check("E8.9 derived features", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E8.10: Summary CSV ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    import csv

    if csv_path and os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            csv_rows = list(reader)

        check("CSV row count matches Parquet",
              len(csv_rows) == hdf5_leaf_count,
              f"csv={len(csv_rows)}, hdf5={hdf5_leaf_count}")

        # Verify key columns present
        csv_headers = csv_rows[0].keys() if csv_rows else set()
        for col in ["experiment_id", "model", "num_qubits", "best_energy",
                     "noise_environment", "placement_qubits"]:
            check(f"CSV has column '{col}'", col in csv_headers)

    else:
        check("Summary CSV exists", False, f"path: {csv_path}")

except Exception as e:
    check("E8.10 summary CSV", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"E8 VALIDATION RESULTS: {passed} passed, {failed} failed")
print(f"{'='*70}")

if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  - {e}")

if failed == 0:
    print("\nE8 VALIDATION: ALL CHECKS PASSED")
    sys.exit(0)
else:
    print(f"\nE8 VALIDATION: {failed} CHECKS FAILED")
    sys.exit(1)
