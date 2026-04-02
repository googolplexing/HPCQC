#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase E — E4: Multi-Calibration Twin Simulation Battery.

Tests the twin simulator: all 11 noise environments per placement,
noiseless deduplication across calibrations, and calibration sensitivity.

VE7:  All 11 environments produce results for each placement.
VE8:  Noiseless results identical across calibrations (deduplication).
VE9:  noise_full energy differs between calibrations.

Run on LUMI standard partition (CPU only, 4q circuits):
    srun ... python tests/e4_twin_sim_validation.py

Expected: E4 VALIDATION: ALL CHECKS PASSED

RED-SPEC-002 §4 — Multi-Calibration Twin Simulation Battery
RED-DIRECTIVE-E4-SCHEMA-v1.0
"""

import sys
import os
import json
import time
import traceback

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

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


# ══════════════════════════════════════════════════════════════════════
print("\n=== E4.1: Noise Config Definitions ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.noise_configs import (
        NOISE_ENVIRONMENTS, NOISE_ENV_BY_NAME,
        NOISELESS_ENVS, NOISY_ENVS, TIER_A_ENVS, TIER_B_ENVS,
        NoiseConfig, get_active_channels_string, get_env_names,
    )

    check("11 noise environments defined",
          len(NOISE_ENVIRONMENTS) == 11,
          f"got {len(NOISE_ENVIRONMENTS)}")

    check("2 noiseless environments", len(NOISELESS_ENVS) == 2)
    check("5 Tier A environments", len(TIER_A_ENVS) == 5)
    check("3 Tier B environments", len(TIER_B_ENVS) == 3)
    check("1 noise_full environment",
          "noise_full" in NOISE_ENV_BY_NAME)

    # Verify tiered measurement stats intervals
    check("Tier A interval = 5",
          all(nc.measurement_stats_interval == 5 for nc in TIER_A_ENVS),
          f"intervals: {[nc.measurement_stats_interval for nc in TIER_A_ENVS]}")
    check("Tier B interval = 20",
          all(nc.measurement_stats_interval == 20 for nc in TIER_B_ENVS),
          f"intervals: {[nc.measurement_stats_interval for nc in TIER_B_ENVS]}")
    check("noise_full interval = 10",
          NOISE_ENV_BY_NAME["noise_full"].measurement_stats_interval == 10)
    check("Noiseless interval = 0 (disabled)",
          all(nc.measurement_stats_interval == 0 for nc in NOISELESS_ENVS))

    # Verify noiseless use statevector
    check("Noiseless use statevector method",
          all(nc.method == "statevector" for nc in NOISELESS_ENVS))
    check("Noisy use density_matrix method",
          all(nc.method == "density_matrix" for nc in NOISY_ENVS))

    # Verify environment names
    names = get_env_names()
    check("All 11 names unique",
          len(set(names)) == 11,
          f"names: {names}")

    print(f"    Environments: {names}")

except Exception as e:
    check("E4.1 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E4.2: Placement Noise Model Builder ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.twin_simulator import build_placement_noise_model

    # Load Q50 calibration
    cal_path = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")
    with open(cal_path) as f:
        cal_data = json.load(f)

    # Build noise model for specific qubits (4q placement)
    test_qubits = ["QB6", "QB7", "QB13", "QB12"]
    nm, cm = build_placement_noise_model(
        cal_data, test_qubits,
        {"single_qubit_depolarizing": True, "two_qubit_depolarizing": True,
         "t1_relaxation": True, "t2_dephasing": True, "readout_error": True},
    )

    check("Full noise model created", nm is not None)
    check("Coupling map created", cm is not None)

    # Noiseless returns None
    nm_none, cm_none = build_placement_noise_model(cal_data, test_qubits, None)
    check("Noiseless: noise_model is None", nm_none is None)
    check("Noiseless: coupling_map is None", cm_none is None)

    # Individual channel builds
    for channel in ["single_qubit_depolarizing", "two_qubit_depolarizing",
                    "t1_relaxation", "t2_dephasing", "readout_error"]:
        channels = {k: (k == channel) for k in [
            "single_qubit_depolarizing", "two_qubit_depolarizing",
            "t1_relaxation", "t2_dephasing", "readout_error",
        ]}
        nm_ch, _ = build_placement_noise_model(cal_data, test_qubits, channels)
        check(f"Individual channel '{channel}' builds",
              nm_ch is not None,
              f"returned None for {channel}")

except Exception as e:
    check("E4.2 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E4.3: VE7 — All 11 Environments Produce Results ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.sweep.twin_simulator import run_twin_battery

    # Build a simple 4q circuit for testing
    qc = QuantumCircuit(4)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)
    qc.cx(2, 3)

    # TFIM 4q Hamiltonian
    terms = []
    for i in range(3):
        zz = ["I"] * 4
        zz[i] = "Z"
        zz[i + 1] = "Z"
        terms.append(("".join(zz), -1.0))
    for i in range(4):
        x = ["I"] * 4
        x[i] = "X"
        terms.append(("".join(x), -1.0))
    observable = SparsePauliOp.from_list(terms)

    print("    Running 11-environment battery on 4q placement...")
    t0 = time.time()
    battery = run_twin_battery(
        circuit=qc,
        observable=observable,
        qubit_names=test_qubits,
        calibration_data=cal_data,
        calibration_id="q50_20260330",
        placement_id="p_0001",
        topology_hash="test_hash_001",
        seed=42,
        device="CPU",
    )
    t_battery = time.time() - t0
    print(f"    Battery complete: {t_battery:.2f}s, "
          f"{battery.simulated_count} simulated, "
          f"{battery.deduplicated_count} deduplicated")

    check("VE7: 11 results returned",
          len(battery.results) == 11,
          f"got {len(battery.results)}")

    # Check each environment produced a result
    env_names_seen = set()
    for r in battery.results:
        env_names_seen.add(r.environment)
        if r.error is not None:
            check(f"VE7: {r.environment} no error", False, r.error)

    check("VE7: all 11 environment names present",
          env_names_seen == set(get_env_names()),
          f"missing: {set(get_env_names()) - env_names_seen}")

    # All environments should have non-None energy
    energies = {r.environment: r.energy for r in battery.results}
    all_have_energy = all(e is not None for e in energies.values())
    check("VE7: all environments have non-None energy",
          all_have_energy,
          f"None energies: {[k for k, v in energies.items() if v is None]}")

    # All energies should be finite
    all_finite = all(np.isfinite(e) for e in energies.values() if e is not None)
    check("VE7: all energies finite", all_finite)

    # Print energy breakdown
    print(f"\n    === Energy Breakdown (4q GHZ on {test_qubits}) ===")
    for r in battery.results:
        dedup = " [DEDUP]" if r.is_deduplicated else ""
        print(f"      {r.environment:25s}: E={r.energy:+.6f}  "
              f"t={r.execution_time_s:.3f}s  tier={r.tier}{dedup}")

except Exception as e:
    check("E4.3 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E4.4: VE8 — Noiseless Deduplication ===")
# ══════════════════════════════════════════════════════════════════════

def _cal_to_dict(base_dict, device_cal):
    """Build a calibration JSON dict from a DeviceCalibration object.

    Overlays the perturbed qubit/gate values onto the base JSON structure
    so the noise model builder can use it directly.
    """
    result = json.loads(json.dumps(base_dict))  # deep copy
    for qname, qcal in device_cal.qubits.items():
        if qname in result.get("qubits", {}):
            result["qubits"][qname]["t1_us"] = qcal.t1_us
            result["qubits"][qname]["t2_us"] = qcal.t2_us
            result["qubits"][qname]["readout_fidelity"] = qcal.readout_fidelity
            result["qubits"][qname]["single_gate_error"] = qcal.single_gate_error
    return result

try:
    from lumi_hpc_qc.sweep.twin_simulator import run_multi_calibration_battery
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter
    from lumi_hpc_qc.plugins.calibration_adapters.synthetic import SyntheticAdapter

    # Load real calibration via adapter
    adapter = IQMv2Adapter()
    real_cal = adapter.load(cal_path)

    # Create synthetic calibration (T1 degraded 30%)
    synth = SyntheticAdapter()
    synth_cal = synth.perturb(real_cal, {
        "scale_t1": 0.5,
        "scale_readout": 0.8,
        "scale_gate_error": 3.0,
        "description": "T1 halved, readout degraded 20%, gate error tripled",
    })

    # Get synthetic calibration as raw dict for noise model builder
    # Rebuild the JSON structure from the DeviceCalibration
    synth_cal_data = _cal_to_dict(cal_data, synth_cal)

    # Run multi-calibration battery
    print("    Running 2-calibration battery (real + synthetic: T1×0.5, readout×0.8, gate_error×3)...")
    calibrations = [
        ("q50_real", cal_data),
        ("q50_synth_degraded", synth_cal_data),
    ]

    multi_results = run_multi_calibration_battery(
        circuit=qc,
        observable=observable,
        qubit_names=test_qubits,
        calibrations=calibrations,
        placement_id="p_0001",
        topology_hash="test_hash_001",
        seed=42,
        device="CPU",
    )

    check("VE8: 2 battery results returned",
          len(multi_results) == 2,
          f"got {len(multi_results)}")

    # First calibration: noiseless computed (simulated)
    first = multi_results[0]
    noiseless_first = [r for r in first.results if r.tier == "noiseless"]
    check("VE8: first calibration computes noiseless",
          all(not r.is_deduplicated for r in noiseless_first),
          "some noiseless were marked as deduplicated in first cal")

    # Second calibration: noiseless should be deduplicated
    second = multi_results[1]
    noiseless_second = [r for r in second.results if r.tier == "noiseless"]
    check("VE8: second calibration deduplicates noiseless",
          all(r.is_deduplicated for r in noiseless_second),
          f"dedup status: {[(r.environment, r.is_deduplicated) for r in noiseless_second]}")

    # Deduplicated energies should be identical
    for env_name in ["noiseless", "topology_noiseless"]:
        e1 = next((r.energy for r in first.results if r.environment == env_name), None)
        e2 = next((r.energy for r in second.results if r.environment == env_name), None)
        if e1 is not None and e2 is not None:
            check(f"VE8: {env_name} identical across calibrations",
                  abs(e1 - e2) < 1e-10,
                  f"real={e1}, synth={e2}, diff={abs(e1-e2):.2e}")

    # Deduplication count
    check("VE8: second battery has 2 deduplicated results",
          second.deduplicated_count == 2,
          f"got {second.deduplicated_count}")

    print(f"    First cal:  {first.simulated_count} simulated, {first.deduplicated_count} dedup")
    print(f"    Second cal: {second.simulated_count} simulated, {second.deduplicated_count} dedup")

except Exception as e:
    check("E4.4 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E4.5: VE9 — Calibration Affects noise_full Energy ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Compare noise_full energy between real and synthetic calibration
    e_real = next((r.energy for r in first.results if r.environment == "noise_full"), None)
    e_synth = next((r.energy for r in second.results if r.environment == "noise_full"), None)

    check("VE9: real calibration has noise_full energy",
          e_real is not None)
    check("VE9: synthetic calibration has noise_full energy",
          e_synth is not None)

    if e_real is not None and e_synth is not None:
        diff = abs(e_real - e_synth)
        check("VE9: noise_full energy differs between calibrations",
              diff > 1e-6,
              f"real={e_real:.6f}, synth={e_synth:.6f}, diff={diff:.2e}")
        print(f"    noise_full real:  {e_real:+.6f}")
        print(f"    noise_full synth: {e_synth:+.6f}")
        print(f"    difference:       {diff:.6f}")

    # Also check that noisy environments differ from noiseless
    e_noiseless = next((r.energy for r in first.results if r.environment == "noiseless"), None)
    if e_noiseless is not None and e_real is not None:
        diff_noisy = abs(e_noiseless - e_real)
        check("VE9: noise_full differs from noiseless",
              diff_noisy > 1e-4,
              f"noiseless={e_noiseless:.6f}, noise_full={e_real:.6f}")

except Exception as e:
    check("E4.5 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E4.6: Noise Environment Ordering and Tags ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Each result should carry the correct environment name and tier
    for r in battery.results:
        env_config = NOISE_ENV_BY_NAME.get(r.environment)
        check(f"Tag: {r.environment} has correct tier",
              env_config is not None and r.tier == env_config.tier,
              f"result tier={r.tier}, config tier={env_config.tier if env_config else 'missing'}")

    # Channels active string should be populated
    for r in battery.results:
        check(f"Tag: {r.environment} has channels_active",
              r.noise_channels_active is not None and len(r.noise_channels_active) > 0,
              f"channels_active={r.noise_channels_active}")

except Exception as e:
    check("E4.6 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== E4.7: Reproducibility (same seed = same energy) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    battery2 = run_twin_battery(
        circuit=qc,
        observable=observable,
        qubit_names=test_qubits,
        calibration_data=cal_data,
        calibration_id="q50_20260330",
        placement_id="p_0001",
        topology_hash="test_hash_001",
        seed=42,
        device="CPU",
    )

    for r1, r2 in zip(battery.results, battery2.results):
        if r1.energy is not None and r2.energy is not None:
            diff = abs(r1.energy - r2.energy)
            check(f"Repro: {r1.environment} |ΔE| < 1e-8",
                  diff < 1e-8,
                  f"diff={diff:.2e}")

except Exception as e:
    check("E4.7 block", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"E4 VALIDATION: {passed} passed, {failed} failed")
if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  ✗ {e}")
    print(f"\nE4 VALIDATION: FAILED ({failed} failures)")
    sys.exit(1)
else:
    print(f"\nE4 VALIDATION: ALL {passed} CHECKS PASSED")
    sys.exit(0)
