#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Probe: confirm the device-calibrated statevector path never throws
"QuantumError: Kraus is empty" for ANY --noise selection -- in particular the
worst case, --noise=thermal_relaxation_error, where the T1/T2 channel is added
WITHOUT a depolarizing error composed alongside it, so a qubit with T2 > T1
(e.g. QB21 in the q50 calibration) contributes a BARE genuine non-unitary Kraus
map. That bare-Kraus configuration is exactly what crashed the old logical-gates
path; this probe verifies that pinning method="statevector" (which makes Aer run
enable_kraus_method deterministically) keeps it safe.

This is a LOGIN-NODE probe, not a batch job: it runs a couple of tiny circuits
for a handful of shots and costs no allocation. Run it inside the qiskit_aer
container after `git pull`:

    export PROJECT_DIR=$(pwd)
    singularity exec /appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif \
        python3 tests/probe_kraus_statevector.py \
        examples/q50_calibration_20260523_d6ebf808.json

Exit code 0 = every spec ran clean (fix holds). Non-zero = a spec threw; the
traceback and the offending spec are printed.
"""
import os
import sys
import glob
import traceback

import numpy as np

# Make the in-repo package + the repo root importable from a checkout.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for p in (_ROOT, os.path.join(_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from lumi_hpc_qc.backends.noise_spec import parse_noise_spec  # noqa: E402
from lumi_hpc_qc.backends.prepare import prepare_simulation  # noqa: E402
import floquet_runner as fr  # noqa: E402  (for build_circuit / build_init_bit_array)


def _resolve_calibration(argv):
    if len(argv) > 1:
        return argv[1]
    # Fall back to the newest calibration in examples/.
    cands = sorted(glob.glob(os.path.join(_ROOT, "examples",
                                           "q50_calibration_*.json")))
    if not cands:
        sys.exit("ERROR: no calibration given and none found in examples/. "
                 "Pass a calibration JSON as the first argument.")
    return cands[-1]


def _noise_tag(result, idx=0):
    """Best-effort read of the per-experiment noise-sampling tag Aer stamps
    ('kraus' = branch 4 precompute path, 'superop', 'readout', 'ideal', or
    absent = general circuit sampling)."""
    try:
        md = result.results[idx].metadata
        if isinstance(md, dict):
            return md.get("noise", "(none/general)")
        return getattr(md, "noise", "(none/general)")
    except Exception:
        return "(unavailable)"


def _build_small_circuits(num_qubits, num_kicks=2):
    """Two tiny Floquet circuits using the production circuit builder."""
    np.random.seed(0)
    h_x = (1 - 0.03) * np.pi
    init = fr.build_init_bit_array(3, num_qubits)
    Jz = np.random.uniform(-1.5 * np.pi, -0.5 * np.pi, num_qubits)
    hz = np.random.uniform(-np.pi, np.pi, num_qubits)
    return [fr.build_circuit(k, hz, Jz, init, num_qubits, h_x)
            for k in (1, num_kicks)]


def _run_one(spec_str, calibration_path, num_qubits):
    spec = parse_noise_spec(spec_str)
    circuits = _build_small_circuits(num_qubits)
    prep = prepare_simulation(
        circuits, "device-calibrated",
        spec=spec, calibration_path=calibration_path,
        num_qubits=num_qubits, durations=(None, None, None),
        t2_mode="ramsey", optimization_level=3, num_processes=1)
    run_circuits, simulator = prep.run_circuits, prep.simulator
    method = simulator.options.method
    result = simulator.run(run_circuits, shots=8, memory=False).result()
    if not result.success:
        raise RuntimeError(f"result.success is False: {result.status}")
    return method, _noise_tag(result), prep.info.get("selected_qubits", [])


def main():
    calibration_path = _resolve_calibration(sys.argv)
    num_qubits = 10
    if not os.path.isfile(calibration_path):
        sys.exit(f"ERROR: calibration not found: {calibration_path}")

    print("=" * 68)
    print(" probe_kraus_statevector")
    print("=" * 68)
    print(f"calibration : {calibration_path}")
    print(f"num_qubits  : {num_qubits}")
    print("-" * 68)

    # thermal-only is the worst case (bare genuine-Kraus). The other two are
    # controls: 'all' is the production model, 'no-thermal' has no Kraus at all.
    specs = ["thermal_relaxation_error", "all", "1q,2q,measurement"]
    failures = []
    for s in specs:
        try:
            method, tag, qubits = _run_one(s, calibration_path, num_qubits)
            print(f"[PASS] --noise={s:<28} method={method:<11} "
                  f"noise-tag={tag}")
            if s == "thermal_relaxation_error":
                print(f"        (bare-Kraus worst case ran clean; method must "
                      f"be 'statevector', is '{method}')")
        except Exception as e:  # noqa: BLE001
            failures.append(s)
            print(f"[FAIL] --noise={s}")
            print("-" * 68)
            traceback.print_exc()
            print("-" * 68)

    print("-" * 68)
    if failures:
        print(f"RESULT: FAIL ({len(failures)}/{len(specs)} specs threw): "
              f"{failures}")
        sys.exit(1)
    print(f"RESULT: PASS (all {len(specs)} specs ran clean under statevector)")
    sys.exit(0)


if __name__ == "__main__":
    main()
