#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""V19 acceptance test — measurement stats end-to-end.

Implements the 6-step test from RED-RESP-V19-v1.0 §1 Q3:

  Step 1: Run TFIM 2q noisy with capture_measurement_stats=true, interval=2
  Step 2: Verify {exp_id}_measurement_stats.jsonl sidecar exists
  Step 3: Run export_hdf5() on the result
  Step 4: Verify /experiments/{name}/measurement_stats is a string array
          with valid JSON entries containing required fields
  Step 5: Verify measurement_stats attrs: grouping_algorithm, interval, num_entries
  Step 6: Run TFIM 2q noisy WITHOUT capture — verify no sidecar, no HDF5 dataset

Uses TFIM 2q + COBYLA maxiter=3 for speed. Shot-based (4096 shots) with
full Q50 noise model to exercise the complete capture path.

Usage:
    python tests/v19_validation.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback

# Ensure project is on path
project_dir = os.environ.get(
    "PROJECT_DIR",
    os.environ.get(
        "SINGULARITYENV_PROJECT_DIR",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
)
sys.path.insert(0, os.path.join(project_dir, "src"))

PASSED = 0
FAILED = 0
CHECKS = []


def check(name: str, condition: bool, detail: str = ""):
    global PASSED, FAILED
    if condition:
        PASSED += 1
        print(f"  PASS: {name}")
    else:
        FAILED += 1
        msg = f"  FAIL: {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)
    CHECKS.append((name, condition))


def run_vqe_with_config(output_dir: str, capture: bool) -> dict:
    """Run a minimal TFIM 2q noisy VQE and return result data."""
    from lumi_hpc_qc.types import ExperimentConfig, CheckpointConfig

    config = ExperimentConfig(
        model="byo",
        model_params={
            # TFIM 2q: H = -J·ZZ - h·(XI + IX), J=1, h=1
            "pauli_list": [("ZZ", -1.0), ("XI", -1.0), ("IX", -1.0)],
        },
        ansatz="su2",
        ansatz_params={"reps": 1},
        optimizer="cobyla",
        optimizer_params={"maxiter": 3, "rhobeg": 0.5},
        gradient="none",
        initializer="random",
        initializer_params={"seed": 42},
        backend="aer_gpu",
        backend_params={
            "method": "density_matrix",
            "shots": 4096,
            "noise_model_file": "examples/q50_calibration_20260330.json",
            "noise_channels": {
                "single_qubit_depolarizing": True,
                "two_qubit_depolarizing": True,
                "t1_relaxation": True,
                "t2_dephasing": True,
                "readout_error": True,
            },
            "coupling_map_source": "calibration",
        },
        precision="double",
        checkpoint=CheckpointConfig(enabled=False),
        capture_measurement_stats=capture,
        measurement_stats_interval=2,  # capture every 2nd eval for test speed
        output_dir=output_dir,
    )

    from lumi_hpc_qc.orchestration.workflow import VQEWorkflow
    workflow = VQEWorkflow()
    record = workflow.run(config)

    return {
        "experiment_id": record.experiment_id,
        "config": config,
        "record": record,
        "output_dir": output_dir,
    }


def main():
    global PASSED, FAILED

    print("=" * 70)
    print("  V19 ACCEPTANCE TEST — Measurement Stats End-to-End")
    print("  RED-RESP-V19-v1.0 §1 Q3: 6-step validation")
    print("=" * 70)

    with tempfile.TemporaryDirectory(prefix="v19_") as tmpdir:
        # ─── STEP 1: Run with capture enabled ───
        print("\n[Step 1] Running TFIM 2q noisy with capture_measurement_stats=true")
        try:
            result_on = run_vqe_with_config(
                os.path.join(tmpdir, "capture_on"), capture=True
            )
            check("Step 1: VQE completes with capture enabled", True)
        except Exception as e:
            check("Step 1: VQE completes with capture enabled", False,
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()
            print(f"\nV19 VALIDATION: {PASSED} passed, {FAILED} failed (ABORTED)")
            sys.exit(1)

        exp_id = result_on["experiment_id"]
        model_dir = os.path.join(tmpdir, "capture_on", "byo")
        print(f"  Experiment ID: {exp_id}")

        # ─── STEP 2: Verify sidecar JSONL exists ───
        print("\n[Step 2] Checking sidecar JSONL")
        sidecar_path = os.path.join(model_dir, f"{exp_id}_measurement_stats.jsonl")
        sidecar_exists = os.path.exists(sidecar_path)
        check("Step 2a: Sidecar JSONL file exists", sidecar_exists)

        if sidecar_exists:
            with open(sidecar_path) as f:
                sidecar_lines = [line.strip() for line in f if line.strip()]
            check("Step 2b: Sidecar has ≥1 entry", len(sidecar_lines) >= 1,
                  f"got {len(sidecar_lines)} lines")

            # Validate each line is valid JSON with required fields
            required_fields = {"evaluation", "pauli_group", "basis_rotations",
                               "counts", "group_expectation", "shots"}
            all_valid = True
            for i, line in enumerate(sidecar_lines):
                try:
                    entry = json.loads(line)
                    missing = required_fields - set(entry.keys())
                    if missing:
                        all_valid = False
                        print(f"    Line {i}: missing fields: {missing}")
                except json.JSONDecodeError as e:
                    all_valid = False
                    print(f"    Line {i}: invalid JSON: {e}")
            check("Step 2c: All sidecar entries have required fields", all_valid)

            # Check that iteration field is present
            first_entry = json.loads(sidecar_lines[0])
            check("Step 2d: Entry has 'iteration' field",
                  "iteration" in first_entry)
        else:
            check("Step 2b: Sidecar has ≥1 entry", False, "sidecar not found")
            check("Step 2c: All sidecar entries have required fields", False,
                  "sidecar not found")
            check("Step 2d: Entry has 'iteration' field", False,
                  "sidecar not found")

        # Check result JSON references the sidecar
        result_json_path = os.path.join(model_dir, f"{exp_id}_result.json")
        if os.path.exists(result_json_path):
            with open(result_json_path) as f:
                result_data = json.load(f)
            check("Step 2e: Result JSON references sidecar",
                  "measurement_stats_sidecar" in result_data)
        else:
            check("Step 2e: Result JSON references sidecar", False,
                  "result JSON not found")

        # ─── STEP 3: Run export_hdf5() ───
        print("\n[Step 3] Running export_hdf5()")
        hdf5_path = os.path.join(tmpdir, "v19_test.h5")
        try:
            from lumi_hpc_qc.data.export import export_hdf5
            n_written = export_hdf5([result_json_path], hdf5_path)
            check("Step 3: export_hdf5 succeeds", n_written >= 1,
                  f"wrote {n_written} experiments")
        except Exception as e:
            check("Step 3: export_hdf5 succeeds", False,
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()

        # ─── STEP 4: Verify HDF5 measurement_stats dataset ───
        print("\n[Step 4] Checking HDF5 measurement_stats dataset")
        try:
            import h5py
            with h5py.File(hdf5_path, "r") as hf:
                exp_names = list(hf["experiments"].keys())
                check("Step 4a: HDF5 has ≥1 experiment group",
                      len(exp_names) >= 1, f"groups: {exp_names}")

                exp_grp = hf[f"experiments/{exp_names[0]}"]

                # Check human-readable group name format
                name = exp_names[0]
                parts = name.split("-")
                check("Step 4b: Group name is dash-separated with ≥5 parts",
                      len(parts) >= 5,
                      f"name='{name}', parts={len(parts)}")

                # Check group-level attributes
                check("Step 4c: Group has experiment_id attr",
                      "experiment_id" in exp_grp.attrs)
                check("Step 4d: Group has model attr",
                      "model" in exp_grp.attrs)

                # Check measurement_stats dataset
                has_ms = "measurement_stats" in exp_grp
                check("Step 4e: measurement_stats dataset exists", has_ms)

                if has_ms:
                    ms_ds = exp_grp["measurement_stats"]
                    ms_lines = list(ms_ds[:])
                    check("Step 4f: measurement_stats has ≥1 entry",
                          len(ms_lines) >= 1, f"got {len(ms_lines)}")

                    # Validate entries
                    all_valid = True
                    for i, line in enumerate(ms_lines[:5]):  # check first 5
                        try:
                            entry = json.loads(line)
                            missing = required_fields - set(entry.keys())
                            if missing:
                                all_valid = False
                                print(f"    HDF5 entry {i}: missing: {missing}")
                        except (json.JSONDecodeError, TypeError) as e:
                            all_valid = False
                            print(f"    HDF5 entry {i}: invalid JSON: {e}")
                    check("Step 4g: HDF5 entries are valid JSON with required fields",
                          all_valid)
                else:
                    check("Step 4f: measurement_stats has ≥1 entry", False,
                          "dataset not found")
                    check("Step 4g: HDF5 entries are valid JSON with required fields",
                          False, "dataset not found")

        except ImportError:
            check("Step 4: HDF5 checks", False, "h5py not available")
        except Exception as e:
            check("Step 4: HDF5 checks", False, f"{type(e).__name__}: {e}")
            traceback.print_exc()

        # ─── STEP 5: Verify HDF5 measurement_stats attributes ───
        print("\n[Step 5] Checking HDF5 measurement_stats attributes")
        try:
            import h5py
            with h5py.File(hdf5_path, "r") as hf:
                exp_grp = hf[f"experiments/{exp_names[0]}"]
                if "measurement_stats" in exp_grp:
                    ms_ds = exp_grp["measurement_stats"]
                    check("Step 5a: grouping_algorithm attr = 'qwc'",
                          ms_ds.attrs.get("grouping_algorithm") == "qwc",
                          f"got '{ms_ds.attrs.get('grouping_algorithm')}'")
                    check("Step 5b: interval attr present",
                          "interval" in ms_ds.attrs,
                          f"interval={ms_ds.attrs.get('interval')}")
                    check("Step 5c: num_entries attr matches dataset length",
                          ms_ds.attrs.get("num_entries") == len(ms_ds),
                          f"attr={ms_ds.attrs.get('num_entries')}, len={len(ms_ds)}")
                else:
                    for label in ["5a", "5b", "5c"]:
                        check(f"Step {label}", False, "measurement_stats not found")
        except Exception as e:
            check("Step 5: attribute checks", False, f"{type(e).__name__}: {e}")

        # ─── STEP 6: Run WITHOUT capture — verify absence ───
        print("\n[Step 6] Running TFIM 2q noisy with capture_measurement_stats=false")
        try:
            result_off = run_vqe_with_config(
                os.path.join(tmpdir, "capture_off"), capture=False
            )
            check("Step 6a: VQE completes without capture", True)

            exp_id_off = result_off["experiment_id"]
            model_dir_off = os.path.join(tmpdir, "capture_off", "byo")
            sidecar_off = os.path.join(
                model_dir_off, f"{exp_id_off}_measurement_stats.jsonl"
            )
            check("Step 6b: No sidecar JSONL when capture=false",
                  not os.path.exists(sidecar_off))

            # Export to HDF5 and verify no measurement_stats dataset
            hdf5_off = os.path.join(tmpdir, "v19_off.h5")
            result_json_off = os.path.join(
                model_dir_off, f"{exp_id_off}_result.json"
            )
            export_hdf5([result_json_off], hdf5_off)
            import h5py
            with h5py.File(hdf5_off, "r") as hf:
                off_names = list(hf["experiments"].keys())
                off_grp = hf[f"experiments/{off_names[0]}"]
                check("Step 6c: No measurement_stats dataset when capture=false",
                      "measurement_stats" not in off_grp)

        except Exception as e:
            check("Step 6: capture=false verification", False,
                  f"{type(e).__name__}: {e}")
            traceback.print_exc()

        # ─── BONUS: Verify data/tools round-trip ───
        print("\n[Bonus] Testing strip/reconstruct round-trip")
        try:
            import h5py
            import shutil

            # Copy the capture-on HDF5 for round-trip test
            hdf5_rt = os.path.join(tmpdir, "v19_roundtrip.h5")
            shutil.copy2(hdf5_path, hdf5_rt)

            from lumi_hpc_qc.data.tools import (
                strip_basis_rotations,
                reconstruct_basis_rotations,
            )

            # Read original entries
            with h5py.File(hdf5_rt, "r") as hf:
                orig_name = list(hf["experiments"].keys())[0]
                if "measurement_stats" in hf[f"experiments/{orig_name}"]:
                    orig_lines = list(
                        hf[f"experiments/{orig_name}/measurement_stats"][:]
                    )
                else:
                    orig_lines = []

            if orig_lines:
                # Strip
                strip_result = strip_basis_rotations(hdf5_rt)
                check("Bonus a: strip_basis_rotations ran",
                      strip_result["entries_stripped"] > 0,
                      f"stripped {strip_result['entries_stripped']}")

                # Verify stripped
                with h5py.File(hdf5_rt, "r") as hf:
                    stripped_line = json.loads(
                        hf[f"experiments/{orig_name}/measurement_stats"][0]
                    )
                    check("Bonus b: basis_rotations removed after strip",
                          "basis_rotations" not in stripped_line)

                # Reconstruct
                recon_result = reconstruct_basis_rotations(hdf5_rt)
                check("Bonus c: reconstruct_basis_rotations ran",
                      recon_result["entries_reconstructed"] > 0)

                # Verify reconstructed
                with h5py.File(hdf5_rt, "r") as hf:
                    recon_line = json.loads(
                        hf[f"experiments/{orig_name}/measurement_stats"][0]
                    )
                    check("Bonus d: basis_rotations restored after reconstruct",
                          "basis_rotations" in recon_line)
            else:
                print("  Skipped: no measurement_stats entries")

        except Exception as e:
            check("Bonus: strip/reconstruct round-trip", False,
                  f"{type(e).__name__}: {e}")

    # ─── Summary ───
    print("\n" + "=" * 70)
    total = PASSED + FAILED
    if FAILED == 0:
        print(f"  V19 VALIDATION: ALL {total} CHECKS PASSED ✓")
    else:
        print(f"  V19 VALIDATION: {PASSED}/{total} passed, {FAILED} FAILED ✗")
        for name, ok in CHECKS:
            if not ok:
                print(f"    FAILED: {name}")
    print("=" * 70)

    sys.exit(0 if FAILED == 0 else 1)


if __name__ == "__main__":
    main()
