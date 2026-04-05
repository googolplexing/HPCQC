#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E — E10: Validation + FiQCI Examples.

Comprehensive end-to-end validation of the complete Phase E sweep pipeline.
This is the v1.1.0rc1 gate — all remaining VE criteria must pass here.

Validation targets:
  VE14: FiQCI GHZ 3q circuit loaded from QPY, swept across placements,
        results in HDF5 with correct physics (⟨Z⊗3⟩ = 0 for odd-qubit GHZ)
  VE24: Topology columns in production-style sweep output
  VE25: Multi-calibration sweep (real + synthetic) produces distinct results

End-to-end tests:
  - FiQCI circuit builders → QPY → circuit_loader → eval_runner
  - BYO circuit → E1 placement → E4 twin battery → E3 HDF5 → E8 Parquet
  - Synthetic calibration (E9) → sweep → different noise fingerprints
  - Sweep engine regression (TFIM, small)
  - Full export pipeline: 67 columns, fingerprinting populated for noisy envs

Run on LUMI standard partition:
    srun ... python tests/e10_validation.py

Expected: E10 VALIDATION: ALL CHECKS PASSED

RED-SPEC-002 §16
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


cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
if not os.path.exists(cal_path):
    cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260326.json")
assert os.path.exists(cal_path), f"No calibration file found"

test_dir = tempfile.mkdtemp(prefix="e10_test_")


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.1: FiQCI Circuit Builders ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    sys.path.insert(0, os.path.join(project_dir, "examples"))
    from fiqci.build_fiqci_circuits import (
        build_ghz, build_bell, build_star_entanglement, CIRCUITS,
    )

    ghz3 = build_ghz(3)
    check("GHZ-3q has 3 qubits", ghz3.num_qubits == 3)
    check("GHZ-3q has H + 2 CX gates", ghz3.size() == 3)
    check("GHZ-3q is not parameterized", ghz3.num_parameters == 0)

    bell = build_bell()
    check("Bell-2q has 2 qubits", bell.num_qubits == 2)
    check("Bell-2q is not parameterized", bell.num_parameters == 0)

    star4 = build_star_entanglement(4)
    check("Star-4q has 4 qubits", star4.num_qubits == 4)
    # Star: H + 3 CX = 4 gates
    check("Star-4q has H + 3 CX gates", star4.size() == 4)

    check("5 circuits in CIRCUITS dict", len(CIRCUITS) == 5)

except Exception as e:
    check("E10.1 circuit builders", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.2: QPY Round-Trip via Circuit Loader ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from qiskit.qpy import dump, load
    from lumi_hpc_qc.sweep.circuit_loader import load_circuit

    # Save GHZ-3q as QPY
    qpy_dir = os.path.join(test_dir, "fiqci_qpy")
    os.makedirs(qpy_dir, exist_ok=True)
    ghz3_qpy = os.path.join(qpy_dir, "ghz_3q.qpy")
    with open(ghz3_qpy, "wb") as f:
        dump([ghz3], f)
    check("QPY file written", os.path.exists(ghz3_qpy))

    # Load via circuit_loader (E5)
    loaded = load_circuit(qpy_file=ghz3_qpy)
    check("VE14: GHZ-3q loaded from QPY", loaded.circuit is not None)
    check("VE14: loaded circuit has 3 qubits", loaded.num_qubits == 3)
    check("VE14: loaded circuit is not parameterized", not loaded.is_parameterized)
    check("VE14: connectivity extracted",
          len(loaded.connectivity) > 0,
          f"got {loaded.connectivity}")

    # Save Bell-2q and Star-4q
    bell_qpy = os.path.join(qpy_dir, "bell_2q.qpy")
    with open(bell_qpy, "wb") as f:
        dump([bell], f)
    loaded_bell = load_circuit(qpy_file=bell_qpy)
    check("Bell-2q loaded from QPY", loaded_bell.num_qubits == 2)

    star_qpy = os.path.join(qpy_dir, "star_4q.qpy")
    with open(star_qpy, "wb") as f:
        dump([star4], f)
    loaded_star = load_circuit(qpy_file=star_qpy)
    check("Star-4q loaded from QPY", loaded_star.num_qubits == 4)

except Exception as e:
    check("E10.2 QPY round-trip", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.3: VE14 — FiQCI GHZ BYO Sweep ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery
    from lumi_hpc_qc.sweep.noise_configs import NOISE_ENVIRONMENTS, NOISE_ENV_BY_NAME
    from lumi_hpc_qc.data.hdf5_writer import SweepHDF5Writer, SweepResultEntry
    from lumi_hpc_qc.sweep.sweep_engine import _compute_fingerprint, _extract_edge_fidelities
    from qiskit.quantum_info import SparsePauliOp

    # Load calibration
    adapter = IQMv2Adapter()
    device_cal = adapter.load(cal_path)
    with open(cal_path) as f:
        cal_json = json.load(f)

    # Find placements for GHZ-3q (chain connectivity: 0-1, 1-2)
    solver = GeneralPlacementSolver()
    solver.add_device(device_cal)
    ghz_placements = solver.find_all_placements(
        circuit_edges=loaded.connectivity,
        circuit_qubits=3,
    )
    check("VE14: placements found for GHZ-3q on Q50",
          len(ghz_placements) > 0,
          f"found {len(ghz_placements)}")

    # Use top 5 placements for speed
    test_placements = ghz_placements[:5]

    # Observable: Z⊗3 (GHZ parity)
    obs_zzz = SparsePauliOp.from_list([("ZZZ", 1.0)])

    # Run twin battery on first placement, all 11 envs
    p0 = test_placements[0]
    qnames = [p0.qubit_mapping[i] for i in range(3)]

    battery = run_twin_battery(
        circuit=ghz3,
        observable=obs_zzz,
        qubit_names=qnames,
        calibration_data=cal_json,
        calibration_id="cal_real",
        placement_id="_".join(qnames),
        topology_hash=p0.topology_hash,
        environments=NOISE_ENVIRONMENTS,
        seed=42,
        device="CPU",
    )

    check("VE14: twin battery produced 11 results",
          len(battery.results) == 11,
          f"got {len(battery.results)}")

    # Physics check: noiseless GHZ-3q ⟨Z⊗3⟩ = 0 (odd-qubit parity)
    noiseless_result = [r for r in battery.results if r.environment == "noiseless"]
    if noiseless_result:
        noiseless_energy = noiseless_result[0].energy
        check("VE14 Physics: noiseless GHZ-3q ⟨ZZZ⟩ = 0 (odd parity)",
              noiseless_energy is not None and abs(noiseless_energy) < 1e-6,
              f"got {noiseless_energy}")

    # Physics check: noise_full should differ from noiseless
    noise_full_result = [r for r in battery.results if r.environment == "noise_full"]
    if noise_full_result and noiseless_result:
        nf_energy = noise_full_result[0].energy
        check("VE14 Physics: noise_full differs from noiseless",
              nf_energy is not None and noiseless_energy is not None
              and abs(nf_energy - noiseless_energy) > 0.001,
              f"noiseless={noiseless_energy}, noise_full={nf_energy}")

    # Write all results to HDF5
    hdf5_path = os.path.join(test_dir, "fiqci_ghz3.h5")
    with SweepHDF5Writer(hdf5_path, sweep_attrs={"sweep_id": "ve14_ghz3"}) as writer:
        for p in test_placements:
            qn = [p.qubit_mapping[i] for i in range(3)]
            bat = run_twin_battery(
                circuit=ghz3, observable=obs_zzz,
                qubit_names=qn, calibration_data=cal_json,
                calibration_id="cal_real", placement_id="_".join(qn),
                topology_hash=p.topology_hash,
                environments=[NOISE_ENV_BY_NAME["noiseless"],
                              NOISE_ENV_BY_NAME["noise_full"]],
                seed=42, device="CPU",
            )
            for tr in bat.results:
                if tr.error is not None:
                    continue
                energy_val = tr.energy if tr.energy is not None else 0.0
                fp = _compute_fingerprint(tr.counts, 3)
                edge_fid = _extract_edge_fidelities(p, device_cal)
                entry = SweepResultEntry(
                    device_id=device_cal.device_id,
                    device_prefix=device_cal.device_prefix,
                    seed=0, placement_qubits=qn,
                    calibration_id="cal_real",
                    noise_config=tr.environment,
                    energy_trajectory=[energy_val],
                    best_energy=energy_val,
                    total_iterations=1, converged=True,
                    circuit_metrics={"num_qubits": 3,
                                     "topology_name": "3q_chain",
                                     "hamiltonian": "ghz_characterization"},
                    per_qubit_calibration=p.per_qubit_calibration,
                    placement_score=p.score,
                    topology_hash=p.topology_hash,
                    wall_time_seconds=tr.execution_time_s,
                    framework_version="1.1.0-rc1",
                    experiment_id=f"ve14_{'_'.join(qn)}_{tr.environment}",
                    noise_fingerprint=fp,
                    per_edge_cz_fidelity=edge_fid,
                )
                writer.write(entry)

    check("VE14: HDF5 file written", os.path.exists(hdf5_path))

    # Verify HDF5 content
    h5 = h5py.File(hdf5_path, "r")
    leaf_count = [0]
    def _count_leaves(name, obj):
        if isinstance(obj, h5py.Group) and "energy_trajectory" in obj:
            leaf_count[0] += 1
    h5.visititems(_count_leaves)
    check("VE14: HDF5 has result groups",
          leaf_count[0] > 0,
          f"found {leaf_count[0]}")
    h5.close()

except Exception as e:
    check("VE14 FiQCI GHZ sweep", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.4: Full Pipeline — BYO → HDF5 → Parquet ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.data.sweep_export import export_sweep_to_parquet
    import pyarrow.parquet as pq

    parquet_path = os.path.join(test_dir, "fiqci_ghz3.parquet")
    export_result = export_sweep_to_parquet(hdf5_path, parquet_path)

    check("Pipeline: Parquet exported",
          export_result["total_rows"] > 0)
    check("Pipeline: 67 columns",
          export_result["columns"] == 67)
    check("Pipeline: row count matches HDF5",
          export_result["total_rows"] == leaf_count[0])

    table = pq.read_table(parquet_path)

    # Verify BYO circuit metadata flows through
    models = set(table.column("model").to_pylist())
    check("Pipeline: model is 'ghz_characterization'",
          "ghz_characterization" in models,
          f"models: {models}")

    # Verify noise_full rows have fingerprinting populated
    envs = table.column("noise_environment").to_pylist()
    entropies = table.column("measurement_entropy").to_pylist()

    noisy_indices = [i for i, e in enumerate(envs) if e == "noise_full"]
    if noisy_indices:
        sample_entropy = entropies[noisy_indices[0]]
        check("Pipeline: noise_full has measurement_entropy populated",
              sample_entropy is not None and sample_entropy > 0,
              f"got {sample_entropy}")

    # Verify noiseless rows have null fingerprinting
    noiseless_indices = [i for i, e in enumerate(envs) if e == "noiseless"]
    if noiseless_indices:
        noiseless_entropy = entropies[noiseless_indices[0]]
        check("Pipeline: noiseless has null measurement_entropy",
              noiseless_entropy is None,
              f"got {noiseless_entropy}")

    # Verify per_edge_cz_fidelity populated
    cz_col = table.column("per_edge_cz_fidelity").to_pylist()
    if noisy_indices:
        sample_cz = cz_col[noisy_indices[0]]
        check("Pipeline: per_edge_cz_fidelity populated",
              sample_cz is not None and len(sample_cz) > 0,
              f"got {sample_cz}")

except Exception as e:
    check("E10.4 full pipeline", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.5: Synthetic Calibration Integration ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.data.tools.perturb_calibration import generate_synthetic

    # Generate synthetic calibration — must perturb channels that affect
    # the GHZ circuit's gates (H, CX). scale_gate_error affects depolarizing
    # on every gate. scale_t1 alone has no effect (no idle/delay gates).
    synth_path = os.path.join(test_dir, "synth_noisy.json")
    generate_synthetic(
        cal_path,
        {"scale_gate_error": 5.0, "scale_readout": 0.8},
        synth_path,
    )

    with open(synth_path) as f:
        synth_cal_json = json.load(f)

    # Run GHZ-3q on same placement with synthetic cal
    p0 = test_placements[0]
    qn = [p0.qubit_mapping[i] for i in range(3)]

    real_bat = run_twin_battery(
        circuit=ghz3, observable=obs_zzz,
        qubit_names=qn, calibration_data=cal_json,
        calibration_id="cal_real", placement_id="_".join(qn),
        topology_hash=p0.topology_hash,
        environments=[NOISE_ENV_BY_NAME["noise_full"]],
        seed=42, device="CPU",
    )

    synth_bat = run_twin_battery(
        circuit=ghz3, observable=obs_zzz,
        qubit_names=qn, calibration_data=synth_cal_json,
        calibration_id="cal_synth_noisy", placement_id="_".join(qn),
        topology_hash=p0.topology_hash,
        environments=[NOISE_ENV_BY_NAME["noise_full"]],
        seed=42, device="CPU",
    )

    real_e = real_bat.results[0].energy
    synth_e = synth_bat.results[0].energy

    check("E9→E10: real calibration produces result", real_e is not None)
    check("E9→E10: synthetic calibration produces result", synth_e is not None)
    check("E9→E10: synthetic produces different energy",
          real_e is not None and synth_e is not None
          and abs(real_e - synth_e) > 0.001,
          f"real={real_e:.6f}, synth={synth_e:.6f}")

    # Verify fingerprints differ
    real_fp = _compute_fingerprint(real_bat.results[0].counts, 3)
    synth_fp = _compute_fingerprint(synth_bat.results[0].counts, 3)

    if real_fp and synth_fp:
        real_ent = real_fp.get("measurement_entropy", 0)
        synth_ent = synth_fp.get("measurement_entropy", 0)
        check("E9→E10: fingerprints differ between calibrations",
              abs(real_ent - synth_ent) > 0.001,
              f"real_entropy={real_ent:.4f}, synth_entropy={synth_ent:.4f}")

except Exception as e:
    check("E10.5 synthetic integration", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.6: VE24 — Topology in Production Sweep ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.sweep_engine import run_sweep_from_dict

    # Small production-style sweep: TFIM 4q, chain+star, 1 seed, 2 envs
    sweep_yaml = {
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "topologies": ["4q_chain", "4q_star"],
                "seeds": 1,
                "noise_configs": ["noiseless", "noise_full"],
                "placement": "top_5",
            }],
            "calibrations": [cal_path],
            "execution": {"cpu_workers": 4},
            "output_dir": os.path.join(test_dir, "ve24_sweep"),
            "sweep_id": "ve24_topology",
        }
    }

    print("  Running topology validation sweep (top_5 placements)...")
    sweep_result = run_sweep_from_dict(sweep_yaml, device="CPU")

    check("VE24: sweep completed without errors",
          sweep_result.total_errors == 0,
          f"{sweep_result.total_errors} errors")

    # Export to Parquet
    pq_path = os.path.join(test_dir, "ve24_sweep", "ve24.parquet")
    exp_result = export_sweep_to_parquet(sweep_result.hdf5_path, pq_path)

    table = pq.read_table(pq_path)
    topos = set(table.column("circuit_topology").to_pylist())
    eq_classes = set(table.column("topology_equivalence_class").to_pylist())

    check("VE24: chain topology in Parquet", "4q_chain" in topos)
    check("VE24: star topology in Parquet", "4q_star" in topos)
    check("VE24: multiple topology equivalence classes",
          len(eq_classes) >= 2,
          f"classes: {eq_classes}")

    # Verify placement_fidelity_score varies across placements
    scores = table.column("placement_fidelity_score").to_pylist()
    unique_scores = len(set(round(s, 6) for s in scores))
    check("VE24: placement scores vary across placements",
          unique_scores > 1,
          f"unique scores: {unique_scores}")

    # Verify fingerprinting populated for noise_full
    envs = table.column("noise_environment").to_pylist()
    entropies = table.column("measurement_entropy").to_pylist()
    noisy_idx = [i for i, e in enumerate(envs) if e == "noise_full"]
    if noisy_idx:
        check("VE24: fingerprinting populated in sweep export",
              entropies[noisy_idx[0]] is not None)

except Exception as e:
    check("VE24 topology in sweep", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.7: VE25 — Multi-Calibration Sweep ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Generate synthetic calibration for second "device"
    synth_cal_path = os.path.join(test_dir, "synth_device.json")
    generate_synthetic(cal_path, {"scale_t1": 0.5, "scale_readout": 0.9},
                       synth_cal_path)

    # Sweep with both calibrations
    multi_cal_yaml = {
        "sweep": {
            "experiments": [{
                "type": "characterization",
                "hamiltonians": ["tfim"],
                "qubit_sizes": [4],
                "topologies": ["4q_chain"],
                "seeds": 1,
                "noise_configs": ["noiseless", "noise_full"],
                "placement": "top_3",
            }],
            "calibrations": [cal_path, synth_cal_path],
            "execution": {"cpu_workers": 4},
            "output_dir": os.path.join(test_dir, "ve25_sweep"),
            "sweep_id": "ve25_multi_cal",
        }
    }

    print("  Running multi-calibration sweep...")
    mc_result = run_sweep_from_dict(multi_cal_yaml, device="CPU")

    check("VE25: multi-calibration sweep completed",
          mc_result.total_errors == 0,
          f"{mc_result.total_errors} errors")

    # Export and verify
    mc_pq = os.path.join(test_dir, "ve25_sweep", "ve25.parquet")
    mc_exp = export_sweep_to_parquet(mc_result.hdf5_path, mc_pq)

    mc_table = pq.read_table(mc_pq)
    cal_sources = set(mc_table.column("calibration_source").to_pylist())
    check("VE25: multiple calibration sources in Parquet",
          len(cal_sources) >= 2,
          f"sources: {cal_sources}")

    # Verify noise_full energies differ between calibrations
    mc_envs = mc_table.column("noise_environment").to_pylist()
    mc_cals = mc_table.column("calibration_source").to_pylist()
    mc_energies = mc_table.column("best_energy").to_pylist()

    cal_energies = {}
    for i, env in enumerate(mc_envs):
        if env == "noise_full":
            cal = mc_cals[i]
            if cal not in cal_energies:
                cal_energies[cal] = []
            cal_energies[cal].append(mc_energies[i])

    if len(cal_energies) >= 2:
        means = {c: np.mean(es) for c, es in cal_energies.items()}
        mean_vals = list(means.values())
        check("VE25: different calibrations produce different mean energies",
              abs(mean_vals[0] - mean_vals[1]) > 0.001,
              f"means: {means}")

except Exception as e:
    check("VE25 multi-calibration", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.8: Bell Pair Physics ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Bell |Φ+⟩: ⟨ZZ⟩ = 1 (perfect correlations)
    obs_zz = SparsePauliOp.from_list([("ZZ", 1.0)])

    # Find 2q placements
    bell_placements = solver.find_all_placements(
        circuit_edges=[(0, 1)], circuit_qubits=2,
    )
    check("Bell: placements found on Q50",
          len(bell_placements) > 0)

    bp = bell_placements[0]
    bqn = [bp.qubit_mapping[i] for i in range(2)]

    bell_bat = run_twin_battery(
        circuit=bell, observable=obs_zz,
        qubit_names=bqn, calibration_data=cal_json,
        calibration_id="cal_real", placement_id="_".join(bqn),
        topology_hash=bp.topology_hash,
        environments=[NOISE_ENV_BY_NAME["noiseless"],
                      NOISE_ENV_BY_NAME["noise_full"]],
        seed=42, device="CPU",
    )

    bell_noiseless = [r for r in bell_bat.results if r.environment == "noiseless"]
    if bell_noiseless:
        bell_e = bell_noiseless[0].energy
        check("Bell Physics: noiseless ⟨ZZ⟩ = 1.0",
              bell_e is not None and abs(bell_e - 1.0) < 1e-6,
              f"got {bell_e}")

    bell_noisy = [r for r in bell_bat.results if r.environment == "noise_full"]
    if bell_noisy:
        bell_ne = bell_noisy[0].energy
        check("Bell Physics: noise_full ⟨ZZ⟩ < 1.0 (noise degrades)",
              bell_ne is not None and bell_ne < 1.0,
              f"got {bell_ne}")

except Exception as e:
    check("E10.8 Bell physics", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.9: Star Topology — Different Connectivity ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Star-4q: hub q[0] connected to q[1], q[2], q[3]
    obs_zzzz = SparsePauliOp.from_list([("ZZZZ", 1.0)])

    star_edges = [(0, 1), (0, 2), (0, 3)]
    star_placements = solver.find_all_placements(
        circuit_edges=star_edges, circuit_qubits=4,
    )
    check("Star-4q: placements found on Q50",
          len(star_placements) > 0,
          f"found {len(star_placements)}")

    # Star topology hash should differ from chain hash
    chain_placements = solver.find_all_placements(
        circuit_edges=[(0,1),(1,2),(2,3)], circuit_qubits=4,
    )
    if star_placements and chain_placements:
        star_hash = star_placements[0].topology_hash
        chain_hash = chain_placements[0].topology_hash
        check("Star vs chain: different topology hashes",
              star_hash != chain_hash,
              f"star={star_hash}, chain={chain_hash}")

    # Star noiseless: ⟨Z⊗4⟩ = 1.0 (even-qubit parity for |0000⟩+|1111⟩)
    sp = star_placements[0]
    sqn = [sp.qubit_mapping[i] for i in range(4)]

    star_bat = run_twin_battery(
        circuit=star4, observable=obs_zzzz,
        qubit_names=sqn, calibration_data=cal_json,
        calibration_id="cal_real", placement_id="_".join(sqn),
        topology_hash=sp.topology_hash,
        environments=[NOISE_ENV_BY_NAME["noiseless"]],
        seed=42, device="CPU",
    )

    star_e = star_bat.results[0].energy
    check("Star Physics: noiseless ⟨ZZZZ⟩ = 1.0 (even parity)",
          star_e is not None and abs(star_e - 1.0) < 1e-6,
          f"got {star_e}")

except Exception as e:
    check("E10.9 star topology", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
print("\n=== E10.10: Regression — All E-Steps ===")
# ═══════════════════════════════════════════════════════════════════════
try:
    # Quick smoke test of each subsystem

    # E1: placement solver
    check("Regression E1: solver returns placements",
          len(chain_placements) > 300)

    # E2: execution planner
    from lumi_hpc_qc.sweep.execution_planner import select_backend
    check("Regression E2: 4q routes to CPU",
          select_backend(4) == "aer_cpu")
    check("Regression E2: 12q routes to GPU",
          select_backend(12) == "aer_gpu")

    # E3: HDF5 writer
    check("Regression E3: HDF5 written in E10.3",
          os.path.exists(hdf5_path))

    # E4: twin simulator
    check("Regression E4: battery returned 11 results",
          len(battery.results) == 11)

    # E5: circuit loader
    check("Regression E5: QPY loading works",
          loaded.num_qubits == 3)

    # E6a: packing
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver as GPS
    rounds = solver.pack_rounds(chain_placements[:20], packing_seed=42)
    check("Regression E6a: packing produces rounds",
          len(rounds) > 0)

    # E7: sweep engine
    check("Regression E7: VE24 sweep completed",
          sweep_result.total_errors == 0)

    # E8: export
    check("Regression E8: Parquet exported",
          os.path.exists(pq_path))

    # E9: synthetic calibration
    check("Regression E9: synthetic cal generated",
          os.path.exists(synth_cal_path))

except Exception as e:
    check("E10.10 regression", False, traceback.format_exc())


# ═══════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════
print(f"\n{'='*70}")
print(f"E10 VALIDATION RESULTS: {passed} passed, {failed} failed")
print(f"{'='*70}")

if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  - {e}")

if failed == 0:
    print("\nE10 VALIDATION: ALL CHECKS PASSED")
    print("\nv1.1.0rc1 GATE: PASSED — all VE criteria satisfied (except VE20/E6b)")
    sys.exit(0)
else:
    print(f"\nE10 VALIDATION: {failed} CHECKS FAILED")
    sys.exit(1)
