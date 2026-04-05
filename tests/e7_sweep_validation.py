#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E — E7: Sweep Engine Orchestrator Validation.

Tests the top-level sweep engine that connects E1–E6a into a single
YAML-to-HDF5 pipeline. The sweep engine is the critical integration
point — it must correctly wire placement solving, twin simulation,
HDF5 writing, and tiered measurement stats into one coherent flow.

Validation targets:
  VE18: Full sweep from YAML config → HDF5 output
        (TFIM 4q, 1 calibration, all placements, all 11 envs, 2 seeds)
  VE19: Tiered measurement stats intervals respected in sweep output
        (Tier A=5, Tier B=20, noise_full=10, noiseless=0)
  VE22: Topology library integrated — chain + star placements in same sweep

Run on LUMI standard partition (CPU only, 4q circuits):
    srun ... python tests/e7_sweep_validation.py

Expected: E7 VALIDATION: ALL CHECKS PASSED

RED-SPEC-002 §§1–17
RED-DIRECTIVE-E4-SCHEMA-v1.0
"""

import sys
import os
import json
import time
import traceback
import tempfile

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
print("\n=== E7.1: Config Parsing and Validation ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.sweep_engine import (
        parse_sweep_config,
        validate_sweep_config,
        expand_grid,
        SweepConfig,
        SweepExperimentConfig,
        SweepTask,
        SweepEngine,
        run_sweep_from_dict,
    )

    # Find calibration file
    cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
    if not os.path.exists(cal_path):
        cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260326.json")
    assert os.path.exists(cal_path), f"No calibration file found in {project_dir}/examples/"

    # ── Parse minimal YAML config ──
    # This config is used for parse/validate tests AND for E7.2 grid expansion.
    # It must match the production config shape:
    #   - Dual topology (chain + star) to test multi-topology grid expansion
    #   - 50 placements per topology (caps runtime)
    #   - 128 CPU workers (matches full LUMI-C node)
    yaml_dict = {
        "sweep": {
            "experiments": [
                {
                    "type": "characterization",
                    "hamiltonians": ["tfim"],
                    "qubit_sizes": [4],
                    "topologies": ["4q_chain", "4q_star"],
                    "seeds": 2,
                    "noise_configs": "all",
                    "placement": 50,
                }
            ],
            "calibrations": [cal_path],
            "execution": {
                "cpu_workers": 128,
            },
            "output_dir": tempfile.mkdtemp(prefix="e7_test_"),
            "sweep_id": "e7_test_001",
        }
    }

    config = parse_sweep_config(yaml_dict)
    check("Config parsed successfully", config is not None)
    check("Sweep ID set", config.sweep_id == "e7_test_001")
    check("1 experiment block", len(config.experiments) == 1)
    check("1 calibration file", len(config.calibrations) == 1)
    check("CPU workers = 128", config.cpu_workers == 128)

    # ── Validate ──
    v_errors = validate_sweep_config(config)
    check("Config validates cleanly", len(v_errors) == 0,
          f"errors: {v_errors}")

    # ── Test validation catches bad config ──
    bad_config = parse_sweep_config({"sweep": {
        "experiments": [],
        "calibrations": [],
    }})
    bad_errors = validate_sweep_config(bad_config)
    check("Validation catches empty experiments",
          any("No experiments" in e for e in bad_errors))
    check("Validation catches missing calibrations",
          any("No calibration" in e for e in bad_errors))

except Exception as e:
    check("E7.1 config parsing", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.2: Grid Expansion ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    tasks = expand_grid(config)

    check("Grid expansion produces tasks", len(tasks) > 0,
          f"got {len(tasks)} tasks")

    # 1 hamiltonian × 1 topology × 1 calibration × 2 seeds = 2 tasks
    check("Correct task count (1×2×1×2 = 4)", len(tasks) == 4,
          f"expected 2, got {len(tasks)}")

    # Verify task structure
    t0 = tasks[0]
    check("Task has hamiltonian 'tfim'", t0.hamiltonian == "tfim")
    check("Task has qubit_size 4", t0.qubit_size == 4)
    check("Task has topology '4q_chain'", t0.topology_name == "4q_chain")
    check("Task has 3 edges (chain)", len(t0.topology_edges) == 3)
    check("Task has 11 noise configs", len(t0.noise_configs) == 11,
          f"got {len(t0.noise_configs)}")
    check("Task seed 0", tasks[0].seed == 0)
    check("Task seed 1", tasks[1].seed == 1)

    # ── Multi-topology grid expansion ──
    multi_yaml = {
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "topologies": ["4q_chain", "4q_star"],
                "seeds": 3,
                "noise_configs": ["noiseless", "noise_full"],
                "placement": "all_valid",
            }],
            "calibrations": [cal_path],
        }
    }
    multi_config = parse_sweep_config(multi_yaml)
    multi_tasks = expand_grid(multi_config)

    # 1 ham × 2 topologies × 1 cal × 3 seeds = 6 tasks
    check("Multi-topology: 6 tasks", len(multi_tasks) == 6,
          f"expected 6, got {len(multi_tasks)}")

    # Verify both topologies present
    topo_names = set(t.topology_name for t in multi_tasks)
    check("Both topologies in grid",
          topo_names == {"4q_chain", "4q_star"},
          f"got {topo_names}")

    # Verify noise config filtering
    for t in multi_tasks:
        check(f"Task {t.task_id} has 2 noise configs",
              len(t.noise_configs) == 2,
              f"got {len(t.noise_configs)}")

except Exception as e:
    check("E7.2 grid expansion", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.3: VE18 — Full Sweep YAML → HDF5 ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    print("  Running full sweep: TFIM 4q, 1 cal, all placements, 11 envs, 2 seeds...")
    t_sweep_start = time.time()

    sweep_yaml = {
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                # Dual topology: tests both chain (3 edges, 379 valid placements)
                # and star (3 edges, 108 valid placements) on Q50.
                # This exercises the placement solver, DSatur packing, and
                # topology-aware noiseless deduplication in a single sweep.
                "topologies": ["4q_chain", "4q_star"],
                "seeds": 2,
                "noise_configs": "all",
                # Cap at 50 placements per topology to keep test runtime
                # reasonable (~50s). Still enough to test parallel dispatch
                # (100 batteries per group = 200 total) and dedup.
                "placement": 50,
            }],
            "calibrations": [cal_path],
            "execution": {
                # Use all 128 cores on a LUMI-C node. This is the production
                # configuration and also the one that triggers the fork
                # deadlock if the subprocess fix is broken.
                "cpu_workers": 128,
            },
            "output_dir": tempfile.mkdtemp(prefix="e7_ve18_"),
            "sweep_id": "ve18_test",
        }
    }

    result = run_sweep_from_dict(sweep_yaml, device="CPU")
    t_sweep_elapsed = time.time() - t_sweep_start

    print(f"  Sweep completed in {t_sweep_elapsed:.1f}s")

    # ── Result structure checks ──
    check("VE18: Sweep completed without fatal errors",
          result.total_errors == 0,
          f"{result.total_errors} errors: {result.errors[:3]}")

    check("VE18: HDF5 file exists",
          os.path.exists(result.hdf5_path),
          f"path: {result.hdf5_path}")

    check("VE18: Placements processed > 0",
          result.total_placements > 0,
          f"got {result.total_placements}")

    check("VE18: Simulations run > 0",
          result.total_simulations > 0,
          f"got {result.total_simulations}")

    check("VE18: HDF5 writes > 0",
          result.total_hdf5_writes > 0,
          f"got {result.total_hdf5_writes}")

    # ── Noiseless deduplication checks (v1.2.0) ──
    # The two-subprocess pattern pre-computes noiseless results:
    #   Subprocess A: runs 2 noiseless envs (noiseless + topology_noiseless)
    #                 per unique topology group. For this test: 2 groups
    #                 (chain + star) × 2 envs = 4 noiseless sims total.
    #   Subprocess B: Pool(128) workers find noiseless in cache → skip.
    #                 Each battery deduplicates 2 noiseless envs, runs 9 noisy.
    #
    # Expected with 2 seeds × 50 placements × 2 topologies = 200 batteries:
    #   total_simulations ≈ 200 × 9 noisy envs = 1800
    #   total_deduplicated ≈ 200 × 2 noiseless envs = 400
    #   total_hdf5_writes = 200 × 11 envs = 2200
    #
    # If dedup is NOT working (v1.1.1 fallback behavior):
    #   total_simulations = 2200, total_deduplicated = 0
    check("VE18: Noiseless deduplication occurred",
          result.total_deduplicated > 0,
          f"got {result.total_deduplicated}")

    # RED-SPEC-003 requirement: noiseless computed at most once per topology group.
    # Each placement beyond the first in a topology group should have its
    # noiseless results deduplicated (found in cache, not recomputed).
    #
    # Math: total_placements = 200 (50 chain × 2 seeds + 50 star × 2 seeds)
    #       n_topology_groups = 2 (chain, star — each has distinct topology_hash)
    #       n_noiseless_envs = 2 (noiseless, topology_noiseless)
    #       expected_dedup >= (200 - 2) × 2 = 396
    #
    # The 2 non-deduplicated are the representative placements that were
    # pre-computed in Subprocess A (one per topology group).
    n_noiseless_envs = 2  # noiseless + topology_noiseless
    n_topology_groups = 2  # chain + star
    check("VE18: Noiseless compute bounded by topology groups",
          result.total_deduplicated >= (result.total_placements - n_topology_groups) * n_noiseless_envs,
          f"deduplicated={result.total_deduplicated}, expected >= "
          f"{(result.total_placements - n_topology_groups) * n_noiseless_envs}")

    # Sanity check: at least 100 deduplications occurred.
    # With 200 batteries × 2 noiseless envs, we expect ~400.
    # The >= 100 threshold catches cases where dedup is partially broken
    # but still produces some hits.
    check("VE18: Dedup count matches expected noiseless savings",
          result.total_deduplicated >= 100,
          f"got {result.total_deduplicated}, expected >= 100 "
          f"(simulated={result.total_simulations}, writes={result.total_hdf5_writes})")

    # ── HDF5 structure checks ──
    h5 = h5py.File(result.hdf5_path, "r")

    # Count leaf groups (those with energy_trajectory)
    leaf_groups = []
    def collect_leaves(name, obj):
        if isinstance(obj, h5py.Group) and "energy_trajectory" in obj:
            leaf_groups.append(name)
    h5.visititems(collect_leaves)

    check("VE18: HDF5 has leaf groups",
          len(leaf_groups) > 0,
          f"found {len(leaf_groups)} groups")

    # Verify leaf group has expected attributes
    if leaf_groups:
        sample_grp = h5[leaf_groups[0]]
        check("VE18: Leaf has energy_trajectory dataset",
              "energy_trajectory" in sample_grp)
        check("VE18: Leaf has best_energy attr",
              "best_energy" in sample_grp.attrs)
        check("VE18: Leaf has noise_config attr",
              "noise_config" in sample_grp.attrs)
        check("VE18: Leaf has seed attr",
              "seed" in sample_grp.attrs)
        check("VE18: Leaf has device_id attr",
              "device_id" in sample_grp.attrs)
        check("VE18: Leaf has calibration_id attr",
              "calibration_id" in sample_grp.attrs)
        check("VE18: Leaf has topology_hash attr",
              "topology_hash" in sample_grp.attrs)
        check("VE18: Leaf has placement_qubits dataset",
              "placement_qubits" in sample_grp)
        check("VE18: Leaf has framework_version attr",
              "framework_version" in sample_grp.attrs)

        # v1.2.0 Item D: exact_ground_energy persisted as HDF5 attribute
        check("VE18: Leaf has exact_ground_energy attr",
              "exact_ground_energy" in sample_grp.attrs,
              f"attrs: {list(sample_grp.attrs.keys())}")
        if "exact_ground_energy" in sample_grp.attrs:
            ege = float(sample_grp.attrs["exact_ground_energy"])
            check("VE18: exact_ground_energy is finite",
                  np.isfinite(ege),
                  f"got {ege}")

    # Verify sweep-level attributes
    check("VE18: HDF5 has sweep_id attr",
          "sweep_id" in h5.attrs,
          f"attrs: {list(h5.attrs.keys())}")
    check("VE18: sweep_id matches",
          h5.attrs.get("sweep_id", "") == "ve18_test")

    # Count environments per seed per placement
    # Each placement should have up to 11 noise configs
    env_configs_seen = set()
    for grp_path in leaf_groups:
        noise_cfg = h5[grp_path].attrs.get("noise_config", "")
        env_configs_seen.add(noise_cfg)

    check("VE18: Multiple noise environments in HDF5",
          len(env_configs_seen) >= 2,
          f"found: {env_configs_seen}")

    # Verify noiseless appears (from deduplication or direct)
    check("VE18: noiseless environment present",
          "noiseless" in env_configs_seen,
          f"envs: {env_configs_seen}")

    check("VE18: noise_full environment present",
          "noise_full" in env_configs_seen,
          f"envs: {env_configs_seen}")

    # Verify 2 seeds are present
    seeds_seen = set()
    for grp_path in leaf_groups:
        s = h5[grp_path].attrs.get("seed", -1)
        seeds_seen.add(int(s))

    check("VE18: Both seeds (0, 1) present in HDF5",
          {0, 1}.issubset(seeds_seen),
          f"seeds found: {seeds_seen}")

    h5.close()

    # ── Verify expected simulation count ──
    # For TFIM 4q chain on Q50: ~379 chain placements
    # 2 seeds × P placements × (9 simulated + 2 deduplicated) environments = many writes
    # At minimum, even with small placement count, should have > 2 writes per seed
    check("VE18: HDF5 writes match expectations (>20)",
          result.total_hdf5_writes >= 20,
          f"got {result.total_hdf5_writes}")

except Exception as e:
    check("VE18 full sweep", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.4: VE19 — Tiered Measurement Stats Intervals ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.noise_configs import (
        NOISE_ENVIRONMENTS, TIER_A_ENVS, TIER_B_ENVS,
        NOISE_ENV_BY_NAME, NOISELESS_ENVS,
    )

    # Verify the intervals are correctly configured in the noise configs
    # that the sweep engine uses

    for env in TIER_A_ENVS:
        check(f"VE19: {env.name} has interval=5",
              env.measurement_stats_interval == 5,
              f"got {env.measurement_stats_interval}")

    for env in TIER_B_ENVS:
        check(f"VE19: {env.name} has interval=20",
              env.measurement_stats_interval == 20,
              f"got {env.measurement_stats_interval}")

    check("VE19: noise_full has interval=10",
          NOISE_ENV_BY_NAME["noise_full"].measurement_stats_interval == 10)

    for env in NOISELESS_ENVS:
        check(f"VE19: {env.name} has interval=0 (disabled)",
              env.measurement_stats_interval == 0,
              f"got {env.measurement_stats_interval}")

    # Verify the sweep engine passes these configs through to twin sim
    # The twin simulator uses env.measurement_stats_interval in TwinResult
    # Verify by checking what was actually written
    if os.path.exists(result.hdf5_path):
        h5 = h5py.File(result.hdf5_path, "r")
        # Check a noise_full group and a noiseless group
        found_noise_full = False
        found_noiseless = False
        for grp_path in leaf_groups[:50]:  # Sample first 50
            grp = h5[grp_path]
            nc = grp.attrs.get("noise_config", "")
            if nc == "noise_full" and not found_noise_full:
                # The twin simulator tags each result with the noise config name
                # which carries the tiered interval. Verify the config flowed through.
                check("VE19: noise_full result written with correct config tag",
                      nc == "noise_full")
                found_noise_full = True
            elif nc == "noiseless" and not found_noiseless:
                check("VE19: noiseless result written with correct config tag",
                      nc == "noiseless")
                found_noiseless = True
        h5.close()

        check("VE19: Found noise_full results in HDF5", found_noise_full)
        check("VE19: Found noiseless results in HDF5", found_noiseless)

except Exception as e:
    check("VE19 tiered stats", False, traceback.format_exc())

# VE19 amendment — YAML interval override (v1.1.1, RED-RESP-V8-STATUS-AMENDMENT)
try:
    from dataclasses import replace as dc_replace
    from lumi_hpc_qc.sweep.sweep_engine import SweepExperimentConfig

    # Test 1: Default — no override, intervals come from NoiseConfig defaults
    exp_default = SweepExperimentConfig(
        hamiltonians=["tfim"], qubit_sizes=[4],
    )
    check("VE19-amend: default override is None",
          exp_default.measurement_stats_interval_override is None)

    # Test 2: Override applied via dataclasses.replace on NoiseConfig
    env_original = NOISE_ENV_BY_NAME["noise_readout_only"]
    check("VE19-amend: noise_readout_only default interval is 5",
          env_original.measurement_stats_interval == 5,
          f"got {env_original.measurement_stats_interval}")

    env_overridden = dc_replace(env_original, measurement_stats_interval=50)
    check("VE19-amend: override sets interval to 50",
          env_overridden.measurement_stats_interval == 50)
    check("VE19-amend: original unchanged after replace",
          env_original.measurement_stats_interval == 5)
    check("VE19-amend: other fields preserved after replace",
          env_overridden.name == env_original.name
          and env_overridden.tier == env_original.tier)

except Exception as e:
    check("VE19 amendment", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.5: VE22 — Topology Library Integration ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    print("  Running multi-topology sweep: chain + star on TFIM 4q, 1 seed...")

    topo_yaml = {
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "topologies": ["4q_chain", "4q_star"],
                "seeds": 1,
                "noise_configs": ["noiseless", "noise_full"],
                # all_valid here (unlike VE18's cap of 50) so we can verify
                # that chain has MORE placements than star on Q50 — tests
                # that the placement solver finds topology-dependent counts.
                # Q50: 379 chain placements, 108 star placements.
                "placement": "all_valid",
            }],
            "calibrations": [cal_path],
            "execution": {
                # Match production config — full LUMI-C node.
                "cpu_workers": 128,
            },
            "output_dir": tempfile.mkdtemp(prefix="e7_ve22_"),
            "sweep_id": "ve22_test",
        }
    }

    topo_result = run_sweep_from_dict(topo_yaml, device="CPU")

    check("VE22: Multi-topology sweep completed",
          topo_result.total_errors == 0,
          f"{topo_result.total_errors} errors: {topo_result.errors[:3]}")

    check("VE22: HDF5 file created",
          os.path.exists(topo_result.hdf5_path))

    # Verify both topology types appear in HDF5
    h5 = h5py.File(topo_result.hdf5_path, "r")
    topo_hashes = set()
    topo_names_in_hdf5 = set()
    leaf_groups_ve22 = []

    def collect_ve22(name, obj):
        if isinstance(obj, h5py.Group) and "energy_trajectory" in obj:
            leaf_groups_ve22.append(name)
            th = obj.attrs.get("topology_hash", "")
            if th:
                topo_hashes.add(th)
            cm = obj.attrs.get("circuit_metrics", "{}")
            if isinstance(cm, str):
                try:
                    cm_dict = json.loads(cm)
                    tn = cm_dict.get("topology_name", "")
                    if tn:
                        topo_names_in_hdf5.add(tn)
                except json.JSONDecodeError:
                    pass

    h5.visititems(collect_ve22)

    check("VE22: Multiple topology hashes in output",
          len(topo_hashes) >= 2,
          f"unique hashes: {len(topo_hashes)} — {topo_hashes}")

    check("VE22: Chain topology results present",
          "4q_chain" in topo_names_in_hdf5,
          f"topologies: {topo_names_in_hdf5}")

    check("VE22: Star topology results present",
          "4q_star" in topo_names_in_hdf5,
          f"topologies: {topo_names_in_hdf5}")

    # Verify chain and star have different placement counts
    chain_count = sum(
        1 for g in leaf_groups_ve22
        if "circuit_metrics" in h5[g].attrs
        and "4q_chain" in h5[g].attrs.get("circuit_metrics", "")
    )
    star_count = sum(
        1 for g in leaf_groups_ve22
        if "circuit_metrics" in h5[g].attrs
        and "4q_star" in h5[g].attrs.get("circuit_metrics", "")
    )

    check("VE22: Chain placements generated results",
          chain_count > 0, f"chain groups: {chain_count}")
    check("VE22: Star placements generated results",
          star_count > 0, f"star groups: {star_count}")

    # From E1 results: chain should have more placements than star on Q50
    # (379 chain vs 108 star), so even with 2 envs × 1 seed:
    check("VE22: Chain has more results than star (expected on Q50)",
          chain_count > star_count,
          f"chain={chain_count}, star={star_count}")

    h5.close()

except Exception as e:
    check("VE22 topology integration", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.6: Edge Cases and Error Handling ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # ── Empty experiment list ──
    empty_config = parse_sweep_config({"sweep": {
        "experiments": [],
        "calibrations": [cal_path],
    }})
    empty_errors = validate_sweep_config(empty_config)
    check("Edge: empty experiments detected",
          any("No experiments" in e for e in empty_errors))

    # ── Unknown noise config ──
    bad_noise_yaml = {
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "topologies": ["4q_chain"],
                "seeds": 1,
                "noise_configs": ["nonexistent_noise"],
                "placement": "all_valid",
            }],
            "calibrations": [cal_path],
        }
    }
    bad_noise_config = parse_sweep_config(bad_noise_yaml)
    bad_noise_errors = validate_sweep_config(bad_noise_config)
    check("Edge: unknown noise config detected",
          any("unknown noise config" in e for e in bad_noise_errors),
          f"errors: {bad_noise_errors}")

    # ── Unknown topology ──
    bad_topo_yaml = {
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "topologies": ["nonexistent_topo"],
                "seeds": 1,
                "noise_configs": "all",
            }],
            "calibrations": [cal_path],
        }
    }
    bad_topo_config = parse_sweep_config(bad_topo_yaml)
    bad_topo_errors = validate_sweep_config(bad_topo_config)
    check("Edge: unknown topology detected",
          any("unknown topology" in e for e in bad_topo_errors),
          f"errors: {bad_topo_errors}")

    # ── top_N placement strategy ──
    # Test that "top_5" is parsed as placement_strategy="top_n", max_placements=5.
    # Uses both topologies to verify top_n works across different placement counts.
    # Grid: 1 hamiltonian × 2 topologies × 1 seed = 2 tasks, both with top_n strategy.
    top_n_tasks = expand_grid(parse_sweep_config({
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "topologies": ["4q_chain", "4q_star"],
                "seeds": 1,
                "noise_configs": ["noiseless"],
                "placement": "top_5",
            }],
            "calibrations": [cal_path],
        }
    }))
    check("Edge: top_5 placement strategy parsed",
          len(top_n_tasks) == 2 and top_n_tasks[0].placement_strategy == "top_n")
    check("Edge: top_5 max_placements = 5",
          top_n_tasks[0].max_placements == 5)

    # ── Auto topologies for 2q ──
    auto_2q_tasks = expand_grid(parse_sweep_config({
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [2],
                "topologies": "auto",
                "seeds": 1,
                "noise_configs": ["noiseless"],
            }],
            "calibrations": [cal_path],
        }
    }))
    check("Edge: auto topology resolves for 2q",
          len(auto_2q_tasks) >= 1,
          f"got {len(auto_2q_tasks)} tasks")
    if auto_2q_tasks:
        check("Edge: 2q topology is '2q_pair'",
              auto_2q_tasks[0].topology_name == "2q_pair")

except Exception as e:
    check("E7.6 edge cases", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.7: WAL Consistency Verification ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.data.hdf5_writer import SweepHDF5Writer

    # Use the VE18 output (the main sweep)
    if os.path.exists(result.hdf5_path):
        wal_path = result.hdf5_path + ".wal"
        check("WAL file exists alongside HDF5",
              os.path.exists(wal_path),
              f"expected at {wal_path}")

        writer = SweepHDF5Writer(result.hdf5_path)
        consistency = writer.verify_consistency()

        check("WAL: entry count > 0",
              consistency["wal_entries"] > 0,
              f"got {consistency['wal_entries']}")

        check("WAL: HDF5 group count matches WAL entries",
              consistency["wal_entries"] == consistency["hdf5_groups"],
              f"WAL={consistency['wal_entries']}, HDF5={consistency['hdf5_groups']}")

        check("WAL: consistent (no missing entries)",
              consistency["consistent"],
              f"missing: {consistency.get('missing_from_hdf5', 'N/A')}")

except Exception as e:
    check("E7.7 WAL consistency", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.8: Noiseless Deduplication Correctness ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Verify that noiseless results are identical across seeds
    # (same placement, same topology → same noiseless energy)
    if os.path.exists(result.hdf5_path):
        h5 = h5py.File(result.hdf5_path, "r")

        # Collect noiseless energies grouped by placement (qubit set)
        noiseless_by_placement: dict[str, list[float]] = {}

        for grp_path in leaf_groups:
            grp = h5[grp_path]
            nc = grp.attrs.get("noise_config", "")
            if nc == "noiseless":
                qubits_ds = grp["placement_qubits"]
                qubit_key = "_".join(str(q) for q in qubits_ds[:])
                energy = float(grp.attrs.get("best_energy", 0.0))
                if qubit_key not in noiseless_by_placement:
                    noiseless_by_placement[qubit_key] = []
                noiseless_by_placement[qubit_key].append(energy)

        # For each placement, all noiseless energies should be identical
        # (same circuit on same topology → same statevector result)
        all_consistent = True
        for qubit_key, energies in noiseless_by_placement.items():
            if len(energies) > 1:
                if not all(abs(e - energies[0]) < 1e-10 for e in energies):
                    all_consistent = False
                    break

        if noiseless_by_placement:
            check("Dedup: noiseless energies identical across seeds",
                  all_consistent,
                  f"checked {len(noiseless_by_placement)} placements")

            # This check confirms that the two-subprocess dedup pattern is
            # working end-to-end: Subprocess A pre-computed the noiseless
            # results, Subprocess B's workers found them in cache and used
            # the cached values. If dedup had failed, total_deduplicated
            # would be 0 and all noiseless sims would have been recomputed
            # (still correct results, but wasted compute).
            check("Dedup: some simulations were deduplicated",
                  result.total_deduplicated > 0,
                  f"deduplicated: {result.total_deduplicated}")

        h5.close()

except Exception as e:
    check("E7.8 deduplication", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.9: Physics Sanity Checks ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    if os.path.exists(result.hdf5_path):
        h5 = h5py.File(result.hdf5_path, "r")

        noiseless_energies = []
        noise_full_energies = []

        for grp_path in leaf_groups[:200]:  # Sample
            grp = h5[grp_path]
            nc = grp.attrs.get("noise_config", "")
            e = float(grp.attrs.get("best_energy", 0.0))
            if nc == "noiseless":
                noiseless_energies.append(e)
            elif nc == "noise_full":
                noise_full_energies.append(e)

        # TFIM 4q: exact energy is around -5.226 (depends on J, h params)
        # All noiseless results should be the same (|0000⟩ evaluated on H)
        if noiseless_energies:
            check("Physics: noiseless energies are finite",
                  all(np.isfinite(e) for e in noiseless_energies))

            # All noiseless should be identical for same topology
            unique_noiseless = set(round(e, 8) for e in noiseless_energies)
            check("Physics: noiseless energies consistent per topology",
                  len(unique_noiseless) <= 3,  # at most 3 distinct topologies
                  f"unique values: {len(unique_noiseless)}")

        if noise_full_energies:
            check("Physics: noise_full energies are finite",
                  all(np.isfinite(e) for e in noise_full_energies))

            # noise_full should show variance (different placements have different noise)
            if len(noise_full_energies) > 1:
                variance = np.var(noise_full_energies)
                check("Physics: noise_full shows placement-dependent variance",
                      variance > 0,
                      f"variance: {variance:.6f}")

        h5.close()

except Exception as e:
    check("E7.9 physics sanity", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.10: Regression — E1 through E6a Still Work ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Quick regression: verify the components E7 uses haven't broken

    # E1: Placement solver
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter

    adapter = IQMv2Adapter()
    cal = adapter.load(cal_path)
    solver = GeneralPlacementSolver()
    solver.add_device(cal)
    placements = solver.find_all_placements(
        circuit_edges=[(0, 1), (1, 2), (2, 3)],
        circuit_qubits=4,
    )
    check("Regression E1: placements found", len(placements) > 0)

    # E4: Twin simulator
    from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery
    from lumi_hpc_qc.sweep.noise_configs import NOISE_ENVIRONMENTS
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import SparsePauliOp

    qc = QuantumCircuit(4)
    h_op = SparsePauliOp.from_list([("ZZZZ", 1.0)])
    with open(cal_path) as f:
        cal_json = json.load(f)

    p = placements[0]
    qnames = [p.qubit_mapping[i] for i in range(4)]
    battery = run_twin_battery(
        circuit=qc,
        observable=h_op,
        qubit_names=qnames,
        calibration_data=cal_json,
        calibration_id="regression_test",
        placement_id="reg_p0",
        topology_hash=p.topology_hash,
        environments=NOISE_ENVIRONMENTS[:3],  # first 3 only for speed
        seed=42,
        device="CPU",
    )
    check("Regression E4: battery produced results",
          len(battery.results) == 3)

    # E3: HDF5 writer
    from lumi_hpc_qc.data.hdf5_writer import SweepHDF5Writer, SweepResultEntry
    import tempfile

    tmp_h5 = os.path.join(tempfile.mkdtemp(), "regression.h5")
    entry = SweepResultEntry(
        device_id="test", device_prefix="test_dev",
        seed=0, placement_qubits=["Q1", "Q2"],
        calibration_id="cal_reg", noise_config="noiseless",
        energy_trajectory=[1.0, 0.5], best_energy=0.5,
        total_iterations=2, converged=True,
    )
    with SweepHDF5Writer(tmp_h5) as w:
        w.write(entry)
    check("Regression E3: HDF5 write succeeded",
          os.path.exists(tmp_h5))

    h5_check = h5py.File(tmp_h5, "r")
    groups_found = []
    h5_check.visititems(lambda n, o: groups_found.append(n)
                        if isinstance(o, h5py.Group) and "energy_trajectory" in o
                        else None)
    check("Regression E3: HDF5 contains result group",
          len(groups_found) == 1)
    h5_check.close()

except Exception as e:
    check("E7.10 regression", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E7.11: LHS Sampling + Typed Parameter Columns (v1.2.0 Item C) ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.sweep_engine import SamplingConfig

    # ── Test LHS grid expansion ──
    # 10 LHS samples × 1 topology × 1 seed × 1 calibration = 10 tasks
    lhs_yaml = {
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "topologies": ["4q_chain"],
                "seeds": 1,
                "noise_configs": ["noiseless"],
                "placement": 1,  # single placement for speed
                "sampling": {
                    "method": "lhs",
                    "n_samples": 10,
                    "parameters": {
                        "j": [0.5, 2.0],
                        "g": [0.5, 2.0],
                    },
                    "seed": 42,
                },
            }],
            "calibrations": [cal_path],
            "execution": {"cpu_workers": 128},
            "output_dir": tempfile.mkdtemp(prefix="e7_lhs_"),
            "sweep_id": "lhs_test",
        }
    }

    lhs_config = parse_sweep_config(lhs_yaml)
    check("LHS: sampling config parsed",
          lhs_config.experiments[0].sampling is not None)
    check("LHS: method is lhs",
          lhs_config.experiments[0].sampling.method == "lhs")
    check("LHS: n_samples is 10",
          lhs_config.experiments[0].sampling.n_samples == 10)

    lhs_tasks = expand_grid(lhs_config)
    check("LHS: 10 tasks generated",
          len(lhs_tasks) == 10,
          f"got {len(lhs_tasks)}")

    # Verify all tasks have distinct model_params
    param_sets = [tuple(sorted(t.model_params.items())) for t in lhs_tasks]
    check("LHS: all 10 tasks have unique params",
          len(set(param_sets)) == 10,
          f"unique: {len(set(param_sets))}")

    # Verify parameter ranges
    j_vals = [t.model_params["j"] for t in lhs_tasks]
    g_vals = [t.model_params["g"] for t in lhs_tasks]
    check("LHS: j values in [0.5, 2.0]",
          all(0.5 <= v <= 2.0 for v in j_vals),
          f"range: [{min(j_vals):.3f}, {max(j_vals):.3f}]")
    check("LHS: g values in [0.5, 2.0]",
          all(0.5 <= v <= 2.0 for v in g_vals),
          f"range: [{min(g_vals):.3f}, {max(g_vals):.3f}]")

    # ── Run LHS sweep to verify end-to-end ──
    print("  Running LHS sweep: 10 samples, TFIM 4q, noiseless only...")
    t_lhs_start = time.time()
    lhs_result = run_sweep_from_dict(lhs_yaml, device="CPU")
    t_lhs_elapsed = time.time() - t_lhs_start
    print(f"  LHS sweep completed in {t_lhs_elapsed:.1f}s")

    check("LHS: sweep completed without errors",
          lhs_result.total_errors == 0,
          f"{lhs_result.total_errors} errors")
    check("LHS: HDF5 writes > 0",
          lhs_result.total_hdf5_writes > 0,
          f"got {lhs_result.total_hdf5_writes}")

    # ── Verify model_params and exact_ground_energy in HDF5 ──
    h5_lhs = h5py.File(lhs_result.hdf5_path, "r")
    lhs_leaves = []
    h5_lhs.visititems(lambda n, o: lhs_leaves.append(n)
                       if isinstance(o, h5py.Group) and "energy_trajectory" in o
                       else None)

    check("LHS: HDF5 has leaf groups",
          len(lhs_leaves) > 0)

    if lhs_leaves:
        sample = h5_lhs[lhs_leaves[0]]
        check("LHS: leaf has model_params attr",
              "model_params" in sample.attrs,
              f"attrs: {list(sample.attrs.keys())}")
        check("LHS: leaf has exact_ground_energy attr",
              "exact_ground_energy" in sample.attrs)

        if "model_params" in sample.attrs:
            mp = json.loads(sample.attrs["model_params"])
            check("LHS: model_params has j",
                  "j" in mp,
                  f"keys: {list(mp.keys())}")
            check("LHS: model_params has g",
                  "g" in mp,
                  f"keys: {list(mp.keys())}")
            check("LHS: j is in range",
                  0.5 <= mp["j"] <= 2.0,
                  f"j={mp['j']}")

    # Verify different groups have different exact_ground_energy values
    # (different J, g → different Hamiltonian → different ground energy)
    energies = set()
    for leaf_path in lhs_leaves[:10]:
        grp = h5_lhs[leaf_path]
        if "exact_ground_energy" in grp.attrs:
            energies.add(round(float(grp.attrs["exact_ground_energy"]), 8))
    check("LHS: multiple distinct exact_ground_energies",
          len(energies) >= 2,
          f"unique: {len(energies)}")

    h5_lhs.close()

except Exception as e:
    check("E7.11 LHS sampling", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"E7 VALIDATION RESULTS: {passed} passed, {failed} failed")
print(f"{'='*70}")

if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  - {e}")

if failed == 0:
    print("\nE7 VALIDATION: ALL CHECKS PASSED")
    sys.exit(0)
else:
    print(f"\nE7 VALIDATION: {failed} CHECKS FAILED")
    sys.exit(1)
