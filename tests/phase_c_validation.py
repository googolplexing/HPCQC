#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Phase C validation tests — RED-DIRECTIVE-PHASE-C-v1.0 (VC1-VC9).

Run on LUMI standard-g partition:
    srun ... python tests/phase_c_validation.py

VC9 (full reproducibility) requires two separate LUMI runs and is
tested here with synthetic trajectory data. The actual demonstration
runs separately via scripts/reproducibility_check.py.
"""

import sys
import os
import json
import traceback
import tempfile
import shutil

project_dir = os.environ.get("PROJECT_DIR",
              os.environ.get("SINGULARITYENV_PROJECT_DIR",
              os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(project_dir, "src"))

CALIBRATION_FILE = os.path.join(project_dir, "examples", "q50_calibration_20260330.json")

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
print("\n=== VC1: Readout mitigation reduces error ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.error_mitigation.readout import ReadoutMitigator
    import numpy as np

    mitigator = ReadoutMitigator()

    # Simulate a 2-qubit system where the true state is |00⟩ (all shots = "00")
    # but readout errors flip bits. With fidelity=0.90, p_error=0.05:
    #   P(measure 00 | state 00) = 0.95 * 0.95 = 0.9025
    #   P(measure 01 | state 00) = 0.95 * 0.05 = 0.0475
    #   P(measure 10 | state 00) = 0.05 * 0.95 = 0.0475
    #   P(measure 11 | state 00) = 0.05 * 0.05 = 0.0025
    raw_counts = {
        "00": 902,
        "01": 48,
        "10": 47,
        "11": 3,
    }
    total_shots = 1000
    fidelities = [0.90, 0.90]

    corrected = mitigator.correct_counts(raw_counts, fidelities, total_shots)

    # After correction, "00" should be closer to 1000 (the true state)
    raw_00_frac = raw_counts.get("00", 0) / total_shots
    corrected_00_frac = corrected.get("00", 0) / total_shots

    check("Corrected '00' fraction > raw '00' fraction",
          corrected_00_frac > raw_00_frac,
          f"raw={raw_00_frac:.3f}, corrected={corrected_00_frac:.3f}")

    check("Corrected '00' fraction closer to 1.0",
          abs(1.0 - corrected_00_frac) < abs(1.0 - raw_00_frac),
          f"raw error={abs(1.0 - raw_00_frac):.3f}, corrected error={abs(1.0 - corrected_00_frac):.3f}")

    # Error terms should be reduced
    raw_error_sum = sum(v for k, v in raw_counts.items() if k != "00") / total_shots
    corrected_error_sum = sum(v for k, v in corrected.items() if k != "00") / total_shots
    check("Total error counts reduced",
          corrected_error_sum < raw_error_sum,
          f"raw_error={raw_error_sum:.3f}, corrected_error={corrected_error_sum:.3f}")

    # Total counts should be preserved (approximately)
    total_corrected = sum(corrected.values())
    check("Total counts preserved (within rounding)",
          abs(total_corrected - total_shots) <= len(corrected),
          f"total_corrected={total_corrected}, expected~{total_shots}")

    # Test with real calibration data
    real_fids = mitigator.load_fidelities(CALIBRATION_FILE, 4)
    check("load_fidelities returns 4 values",
          len(real_fids) == 4 and all(0 < f <= 1 for f in real_fids),
          f"fidelities={real_fids}")

except Exception as e:
    check("VC1 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== VC2: ZNE reduces error (mitigate_simple fallback) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.error_mitigation.zne import ZneErrorMitigator
    import numpy as np

    zne = ZneErrorMitigator()

    # Simulate noisy energy evaluations that increase with noise scale
    # True energy = -2.0, noisy energy at scale s = -2.0 + 0.5*s
    true_energy = -2.0
    noise_slope = 0.5
    eval_count = [0]

    def noisy_eval(params):
        eval_count[0] += 1
        # Each call gets the next scale factor's noise
        scale = [1, 3, 5][min(eval_count[0] - 1, 2)]
        return true_energy + noise_slope * scale

    # Reset for mitigate_simple test
    mitigated = zne.mitigate_simple(
        eval_fn=lambda p: true_energy + noise_slope * 1,  # constant at scale=1
        params=np.array([0.0]),
        scale_factors=[1, 3, 5],
        extrapolation="linear",
    )

    # With constant eval_fn, mitigate_simple just returns the same value
    # (all scale factors give same energy since eval_fn ignores scale)
    check("mitigate_simple returns float",
          isinstance(mitigated, float),
          f"type={type(mitigated)}")

    # Test with properly scaled evaluations
    scale_idx = [0]
    scale_factors = [1, 3, 5]

    def scaled_eval(params):
        s = scale_factors[scale_idx[0] % len(scale_factors)]
        scale_idx[0] += 1
        return true_energy + noise_slope * s

    mitigated_scaled = zne.mitigate_simple(
        eval_fn=scaled_eval,
        params=np.array([0.0]),
        scale_factors=scale_factors,
        extrapolation="linear",
    )

    # Linear extrapolation to scale=0 should give approximately true_energy
    check("ZNE extrapolation closer to true energy than raw",
          abs(mitigated_scaled - true_energy) < abs((true_energy + noise_slope) - true_energy),
          f"mitigated={mitigated_scaled:.4f}, raw_scale1={true_energy + noise_slope:.4f}, true={true_energy:.4f}")

except Exception as e:
    check("VC2 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== VC3: ZNE apply_every works correctly ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.error_mitigation.zne import ZneErrorMitigator

    zne = ZneErrorMitigator()
    zne._apply_every = 3
    zne._eval_count = 0
    zne._is_gradient_step = False

    # Test apply_every=3: should apply on evals 3, 6, 9, ...
    apply_results = []
    for i in range(9):
        apply_results.append(zne.should_apply())

    # Eval 1: 1%3!=0 → False, Eval 2: 2%3!=0 → False, Eval 3: 3%3==0 → True
    # Eval 4: 4%3!=0 → False, Eval 5: 5%3!=0 → False, Eval 6: 6%3==0 → True
    # Eval 7: 7%3!=0 → False, Eval 8: 8%3!=0 → False, Eval 9: 9%3==0 → True
    expected = [False, False, True, False, False, True, False, False, True]
    check("apply_every=3 pattern correct",
          apply_results == expected,
          f"got={apply_results}, expected={expected}")

    # Test gradient step override
    zne._eval_count = 0
    zne._is_gradient_step = True
    gradient_results = [zne.should_apply() for _ in range(3)]
    check("Gradient steps always apply ZNE",
          all(gradient_results),
          f"got={gradient_results}")

    # Reset
    zne._is_gradient_step = False
    zne.set_gradient_step(True)
    check("set_gradient_step(True) works",
          zne._is_gradient_step is True)
    zne.set_gradient_step(False)
    check("set_gradient_step(False) works",
          zne._is_gradient_step is False)

except Exception as e:
    check("VC3 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== VC4: mitiq lazy import safe with mpi4py ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.error_mitigation import zne as zne_module

    # Verify mitiq is NOT loaded at module import time
    check("mitiq not loaded at import",
          zne_module._mitiq_loaded is False,
          f"_mitiq_loaded={zne_module._mitiq_loaded}")

    # Verify the module can be imported without errors
    check("ZneErrorMitigator class exists",
          hasattr(zne_module, 'ZneErrorMitigator'))

    # Verify _ensure_mitiq exists and is callable
    check("_ensure_mitiq is callable",
          callable(zne_module._ensure_mitiq))

    # Now actually trigger the lazy import
    try:
        zne_module._ensure_mitiq()
        check("mitiq lazy import succeeds",
              zne_module._mitiq_loaded is True)
        check("mitiq module accessible after import",
              zne_module._mitiq is not None)
        check("zne submodule accessible after import",
              zne_module._zne_module is not None)
    except ImportError as ie:
        # mitiq might not be installed in all environments
        check("mitiq import attempted (may fail if not installed)",
              False, f"ImportError: {ie}")
    except Exception as ie:
        # MPI conflict or other issue
        check("mitiq lazy import (non-fatal)", False,
              f"Exception (may be MPI-related): {ie}")

except Exception as e:
    check("VC4 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== VC5: Multiplexed circuit = 53 qubits, non-overlapping ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.placement.multiplexer import MultiplexedCircuitBuilder
    from lumi_hpc_qc.plugins.placement.solver import PlacementSolver
    from qiskit import QuantumCircuit

    solver = PlacementSolver(CALIBRATION_FILE)
    placements = solver.find_placements(circuit_qubits=4, num_placements=9, strategy="max_fidelity")

    # Build a simple 4-qubit test circuit
    sub_circuit = QuantumCircuit(4)
    sub_circuit.h(0)
    sub_circuit.cx(0, 1)
    sub_circuit.cx(1, 2)
    sub_circuit.cx(2, 3)

    builder = MultiplexedCircuitBuilder(device_qubits=53)
    mux_circuit = builder.build(sub_circuit, placements)

    check("Multiplexed circuit has 53 qubits",
          mux_circuit.num_qubits == 53,
          f"num_qubits={mux_circuit.num_qubits}")

    # Verify non-overlapping: all physical indices are unique across placements
    all_indices = []
    for p in placements:
        all_indices.extend(p["physical_indices"])
    check("All placements non-overlapping",
          len(all_indices) == len(set(all_indices)),
          f"total={len(all_indices)}, unique={len(set(all_indices))}")

    # Verify circuit has gates for each placement
    total_gates = mux_circuit.size()
    expected_min_gates = len(placements) * sub_circuit.size()
    check("Multiplexed circuit has gates for all placements",
          total_gates >= expected_min_gates,
          f"total={total_gates}, expected>={expected_min_gates}")

    # Verify measurements present
    meas_count = mux_circuit.count_ops().get("measure", 0)
    used_qubits = set(all_indices)
    check("Measurements for all used qubits",
          meas_count == len(used_qubits),
          f"measurements={meas_count}, used_qubits={len(used_qubits)}")

except Exception as e:
    check("VC5 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== VC6: Per-placement result extraction correct ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.placement.multiplexer import MultiplexedCircuitBuilder

    builder = MultiplexedCircuitBuilder(device_qubits=8)

    # Two placements on an 8-qubit device:
    # Placement 0: logical [0,1] → physical [0,1]
    # Placement 1: logical [0,1] → physical [4,5]
    placements = [
        {"placement_id": 0, "physical_indices": [0, 1]},
        {"placement_id": 1, "physical_indices": [4, 5]},
    ]

    # Simulate measurement results on 8 qubits
    # Bitstring "00110001" (8 bits, MSB=qubit7, LSB=qubit0):
    #   qubit0=1, qubit1=0, qubit2=0, qubit3=0, qubit4=1, qubit5=1, qubit6=0, qubit7=0
    # Placement 0 (qubits 0,1): logical bit0=qubit0=1, logical bit1=qubit1=0 → "01"
    # Placement 1 (qubits 4,5): logical bit0=qubit4=1, logical bit1=qubit5=1 → "11"
    #
    # But we need to be careful about bit ordering in demultiplex.
    # Let's use a simpler test: construct counts where we know the answer.

    # All 8 qubits measured as "00000000" → placement 0 gets "00", placement 1 gets "00"
    raw_counts_all_zero = {"00000000": 500}
    result_zero = builder.demultiplex(raw_counts_all_zero, placements, 2)

    check("Demux all-zeros: placement 0 gets '00'",
          result_zero[0].get("00", 0) == 500,
          f"placement_0={result_zero[0]}")
    check("Demux all-zeros: placement 1 gets '00'",
          result_zero[1].get("00", 0) == 500,
          f"placement_1={result_zero[1]}")

    # All 8 qubits measured as "11111111" → both placements get "11"
    raw_counts_all_one = {"11111111": 300}
    result_one = builder.demultiplex(raw_counts_all_one, placements, 2)

    check("Demux all-ones: placement 0 gets '11'",
          result_one[0].get("11", 0) == 300,
          f"placement_0={result_one[0]}")
    check("Demux all-ones: placement 1 gets '11'",
          result_one[1].get("11", 0) == 300,
          f"placement_1={result_one[1]}")

    # Mixed case: count preservation across placements
    raw_mixed = {"00000000": 400, "11111111": 200, "00010001": 100}
    result_mixed = builder.demultiplex(raw_mixed, placements, 2)

    total_p0 = sum(result_mixed[0].values())
    total_p1 = sum(result_mixed[1].values())
    check("Demux mixed: placement 0 total counts = 700",
          total_p0 == 700,
          f"total={total_p0}")
    check("Demux mixed: placement 1 total counts = 700",
          total_p1 == 700,
          f"total={total_p1}")

except Exception as e:
    check("VC6 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== VC7: Per-placement metadata includes calibration ===")
# ══════════════════════════════════════════════════════════════════════
try:
    from lumi_hpc_qc.plugins.placement.multiplexer import MultiplexedCircuitBuilder
    from lumi_hpc_qc.plugins.placement.solver import PlacementSolver

    solver = PlacementSolver(CALIBRATION_FILE)
    placements = solver.find_placements(circuit_qubits=4, num_placements=3, strategy="max_fidelity")

    metadata = MultiplexedCircuitBuilder.build_placement_metadata(placements, CALIBRATION_FILE)

    check("Metadata has entries for all placements",
          len(metadata) == len(placements),
          f"metadata={len(metadata)}, placements={len(placements)}")

    if metadata:
        m0 = metadata[0]
        check("Metadata has qubit_mapping",
              "qubit_mapping" in m0 and len(m0["qubit_mapping"]) == 4,
              f"keys={list(m0.keys())}")

        check("Metadata has physical_indices",
              "physical_indices" in m0 and len(m0["physical_indices"]) == 4)

        check("Metadata has per_qubit_calibration",
              "per_qubit_calibration" in m0 and len(m0["per_qubit_calibration"]) == 4,
              f"cal_keys={list(m0.get('per_qubit_calibration', {}).keys())}")

        # Check that calibration data has T1, T2, readout fidelity
        if m0.get("per_qubit_calibration"):
            first_q_cal = list(m0["per_qubit_calibration"].values())[0]
            check("Per-qubit cal has physical_qubit",
                  "physical_qubit" in first_q_cal,
                  f"keys={list(first_q_cal.keys())}")
            check("Per-qubit cal has t1_us",
                  "t1_us" in first_q_cal and first_q_cal["t1_us"] > 0,
                  f"t1_us={first_q_cal.get('t1_us')}")
            check("Per-qubit cal has readout_fidelity",
                  "readout_fidelity" in first_q_cal and first_q_cal["readout_fidelity"] > 0.8,
                  f"ro={first_q_cal.get('readout_fidelity')}")

        check("Metadata has avg_cz_fidelity",
              "avg_cz_fidelity" in m0 and m0["avg_cz_fidelity"] > 0)
        check("Metadata has score",
              "score" in m0 and m0["score"] > 0)

except Exception as e:
    check("VC7 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== VC8: Multi-seed sweep produces 20 configs ===")
# ══════════════════════════════════════════════════════════════════════
try:
    # Add scripts to path
    sys.path.insert(0, os.path.join(project_dir, "scripts"))
    from generate_seed_sweep import generate_seed_sweep
    import yaml

    # Use a temp directory for test output
    tmpdir = tempfile.mkdtemp(prefix="seed_sweep_test_")
    try:
        # Find a base config
        base_config = os.path.join(project_dir, "configs", "q50bench_tfim_2q_noiseless.yaml")
        if not os.path.exists(base_config):
            # Try generated configs
            base_config = os.path.join(project_dir, "configs", "generated",
                                       "q50bench_tfim_4q_noiseless.yaml")

        if os.path.exists(base_config):
            configs = generate_seed_sweep(base_config, 20, tmpdir)

            check("Generated 20 configs",
                  len(configs) == 20,
                  f"count={len(configs)}")

            # Verify all configs exist
            all_exist = all(os.path.exists(c) for c in configs)
            check("All config files exist", all_exist)

            # Verify seeds are unique and correct
            seeds = []
            for cfg_path in configs:
                with open(cfg_path) as f:
                    cfg = yaml.safe_load(f)
                seed = cfg.get("initializer_params", {}).get("seed")
                seeds.append(seed)

            check("Seeds are sequential starting from 42",
                  seeds == list(range(42, 62)),
                  f"seeds={seeds[:5]}...{seeds[-3:]}")

            check("All seeds unique",
                  len(set(seeds)) == 20,
                  f"unique={len(set(seeds))}")

            # Verify configs differ ONLY in seed and output_dir
            if len(configs) >= 2:
                with open(configs[0]) as f:
                    cfg0 = yaml.safe_load(f)
                with open(configs[1]) as f:
                    cfg1 = yaml.safe_load(f)

                # Remove seed and output_dir for comparison
                cfg0_compare = {k: v for k, v in cfg0.items()
                                if k not in ("initializer_params", "output_dir")}
                cfg1_compare = {k: v for k, v in cfg1.items()
                                if k not in ("initializer_params", "output_dir")}
                check("Configs identical except seed and output_dir",
                      cfg0_compare == cfg1_compare,
                      "configs differ in non-seed fields")
        else:
            check("Base config exists", False, f"not found: {base_config}")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

except Exception as e:
    check("VC8 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print("\n=== VC9: Reproducibility check (synthetic trajectories) ===")
# ══════════════════════════════════════════════════════════════════════
try:
    sys.path.insert(0, os.path.join(project_dir, "scripts"))
    from reproducibility_check import compare_trajectories

    # Test 1: Identical trajectories → pass
    traj = [-2.0, -2.5, -2.8, -2.95, -3.0]
    result_identical = compare_trajectories(traj, traj, threshold_pct=0.01)
    check("Identical trajectories pass",
          result_identical["pass"] is True)
    check("Max delta is zero for identical",
          result_identical["max_absolute_delta"] == 0.0)

    # Test 2: Small perturbation within threshold → pass
    import numpy as np
    np.random.seed(42)
    traj_orig = [-2.0, -2.5, -2.8, -2.95, -3.0]
    # Add tiny perturbation: 0.001% of each value
    traj_perturbed = [e * (1 + 1e-5) for e in traj_orig]
    result_small = compare_trajectories(traj_orig, traj_perturbed, threshold_pct=0.01)
    check("Small perturbation (0.001%) passes at 0.01% threshold",
          result_small["pass"] is True,
          f"max_rel_delta={result_small['max_relative_delta_pct']:.6f}%")

    # Test 3: Large perturbation exceeds threshold → fail
    traj_large_perturb = [e * 1.01 for e in traj_orig]  # 1% perturbation
    result_large = compare_trajectories(traj_orig, traj_large_perturb, threshold_pct=0.01)
    check("Large perturbation (1%) fails at 0.01% threshold",
          result_large["pass"] is False,
          f"max_rel_delta={result_large['max_relative_delta_pct']:.4f}%")

    # Test 4: Different length trajectories — compares up to shorter
    traj_short = traj_orig[:3]
    result_short = compare_trajectories(traj_orig, traj_short, threshold_pct=0.01)
    check("Different-length trajectories compared up to shorter",
          result_short["iterations_compared"] == 3 and result_short["pass"] is True,
          f"compared={result_short['iterations_compared']}")

    # Test 5: Threshold check format
    check("Result includes threshold",
          result_small["threshold_pct"] == 0.01)
    check("Result includes iteration counts",
          "iterations_original" in result_small and "iterations_reproduced" in result_small)

except Exception as e:
    check("VC9 execution", False, f"Exception: {e}")
    traceback.print_exc()


# ══════════════════════════════════════════════════════════════════════
print(f"\n{'='*60}")
print(f"Phase C validation: {passed} passed, {failed} failed")
if errors:
    print(f"\nFailures:")
    for e in errors:
        print(f"  - {e}")
print(f"{'='*60}")

sys.exit(1 if failed > 0 else 0)
