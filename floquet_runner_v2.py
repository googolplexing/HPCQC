#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""
Floquet driver (v2) -- N gate instances in parallel via multiprocessing.

Behaviourally identical to floquet_runner.py, but the per-instance simulation
setup is delegated to the shared seam lumi_hpc_qc.backends.prepare, instead of
an inline noise_source if/elif chain. This is the Stage-1 refactor: the three
noise sources (noiseless / device-calibrated / iqm-fake-backend) now come from
one validated code path that the Phase E sweep engine will also use. The
original floquet_runner.py is kept alongside for A/B output comparison.

ONE Python process, ONE container, ONE srun; parallelism via
multiprocessing.Pool(forkserver).

Noise sources (--noise-source):
  noiseless        : AerSimulator(), no noise, circuit as-written.
  device-calibrated: native-gate decomposition + routing + ALAP scheduling +
                     control/readout noise + gate-duration relaxation (resident)
                     + idle/delay relaxation (custom pass), under statevector.
                     Most faithful; the only mode with idle decoherence.
  iqm-fake-backend : IQM local FakeBackend, static baked noise. Dependency call.

Per-circuit physics (build_circuit, apply_one_floquet_period,
get_autocorrelation) is byte-identical to the original driver.
"""
import os
import sys
import json
import time
import random
import argparse
import traceback
import multiprocessing as mp

import numpy as np

from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

try:
    from lumi_hpc_qc.backends.prepare import prepare_simulation
    HAVE_DEVICE_NOISE = True
except ImportError:
    # Not pip-installed: make the in-repo package importable from a checkout
    # (PROJECT_DIR/src), then retry.
    _proj = os.environ.get("PROJECT_DIR") or os.environ.get("SINGULARITYENV_PROJECT_DIR")
    if _proj:
        sys.path.insert(0, os.path.join(_proj, "src"))
    try:
        from lumi_hpc_qc.backends.prepare import prepare_simulation
        HAVE_DEVICE_NOISE = True
    except ImportError:
        prepare_simulation = None
        HAVE_DEVICE_NOISE = False

# parse_noise_spec has no qiskit dependency; import it independently so the
# --noise flag still validates even if the qiskit-dependent builders are absent.
try:
    from lumi_hpc_qc.backends.noise_spec import parse_noise_spec
except ImportError:
    parse_noise_spec = None

NOISE_SOURCES = ["noiseless", "device-calibrated", "iqm-fake-backend"]


def get_autocorrelation(counts, init_bit_array, num_qubits):
    total_shots = sum(counts.values())
    num_qub = len(list(counts.keys())[0])
    total_corr = 0
    for bitstring, count in counts.items():
        plus = 0
        minus = 0
        bit_array_little = np.array(list(bitstring), dtype=int)
        bit_array = bit_array_little[::-1]
        for wire in range(num_qubits):
            if bit_array[wire] == init_bit_array[wire]:
                plus += 1
            else:
                minus += 1
        temp_corr = (plus - minus) * count
        total_corr += temp_corr
    return total_corr / (total_shots * num_qub)


def apply_one_floquet_period(qc, hz_angles, Jzz_angles, num_qubits, h_x):
    for wire in range(num_qubits):
        qc.rx(h_x, wire)
    for wire in range(num_qubits):
        qc.rz(hz_angles[wire], wire)
    for wire in range(num_qubits - 1):
        qc.rzz(Jzz_angles[wire], wire, wire + 1)


def build_circuit(num_kicks, hz_angles, Jzz_angles, init_bit_array, num_qubits, h_x):
    qc = QuantumCircuit(num_qubits, num_qubits)
    for wire in range(num_qubits):
        if init_bit_array[wire] == 1:
            qc.x(wire)
    for _ in range(num_kicks):
        apply_one_floquet_period(qc, hz_angles, Jzz_angles, num_qubits, h_x)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc


def build_init_bit_array(initial_state, num_qubits):
    init_bit_array = []
    if initial_state == 1:
        for _ in range(num_qubits):
            init_bit_array.append(random.randint(0, 1))
    elif initial_state == 2:
        for wire in range(num_qubits):
            init_bit_array.append(wire % 2)
    elif initial_state == 3:
        for _ in range(num_qubits):
            init_bit_array.append(0)
    else:
        raise ValueError(f"initial_state must be 1-3, got {initial_state}")
    return init_bit_array


def run_one_instance(args_tuple):
    (instance_id, noise_source, calibration_path, output_dir, num_qubits,
     num_shots, num_max_kicks, epsilon, initial_state, t2_mode, durations,
     iqm_device, noise_spec) = args_tuple

    tag = f"[instance {instance_id:02d}]"
    log_path = os.path.join(output_dir, f"instance_{instance_id:02d}.log")
    os.environ["QISKIT_IN_PARALLEL"] = "TRUE"

    log_fh = open(log_path, "w", buffering=1, encoding="utf-8")
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = log_fh
    sys.stderr = log_fh

    try:
        random.seed(instance_id)
        np.random.seed(instance_id)
        h_x = (1 - epsilon) * np.pi
        init_bit_array = build_init_bit_array(initial_state, num_qubits)

        print(f"{tag} noise-source: {noise_source}")
        print(f"{tag} init_bit_array = {init_bit_array}")
        t0 = time.time()

        Jz_angles = np.random.uniform(-1.5 * np.pi, -0.5 * np.pi, num_qubits)
        hz_angles = np.random.uniform(-np.pi, np.pi, num_qubits)
        circuits = [build_circuit(n, hz_angles, Jz_angles, init_bit_array,
                                  num_qubits, h_x) for n in range(num_max_kicks)]

        # Actual durations used (ns). For device-calibrated this is overwritten
        # from the resolved values below; for other sources the CLI override
        # tuple (often all None) is recorded as-is.
        durations_used = {"prx": durations[0], "cz": durations[1],
                          "measure": durations[2]}

        if noise_source == "device-calibrated" and not HAVE_DEVICE_NOISE:
            raise RuntimeError("prepare_simulation not importable")
        if noise_source == "device-calibrated":
            print(f"{tag} transpiling to native gates + scheduling...")
            if noise_spec is not None:
                print(f"{tag} noise channels: {noise_spec.describe()}")
        elif noise_source == "noiseless":
            print(f"{tag} transpiling {len(circuits)} circuits (serial)...")
        elif noise_source == "iqm-fake-backend":
            print(f"{tag} transpiling for IQM fake backend ({iqm_device})...")

        # Single shared preparation seam for all three noise sources
        # (lumi_hpc_qc.backends.prepare). Identical code path to the one the
        # Phase E sweep engine uses.
        prep = prepare_simulation(
            circuits, noise_source,
            spec=noise_spec,
            calibration_path=calibration_path,
            num_qubits=num_qubits,
            durations=durations,
            t2_mode=t2_mode,
            iqm_device=iqm_device,
            optimization_level=3,
            num_processes=1,
        )
        run_circuits = prep.run_circuits
        simulator = prep.simulator

        if noise_source == "device-calibrated":
            dinfo = prep.info
            print(f"{tag} selected qubits: {dinfo['selected_qubits']}")
            print(f"{tag} durations(ns): prx={dinfo['single_gate_time_ns']} "
                  f"cz={dinfo['cz_gate_time_ns']} measure={dinfo['measure_time_ns']} "
                  f"(source: {dinfo['duration_source']})")
            print(f"{tag} t2_mode: {dinfo['t2_mode']}  edges: {dinfo['num_edges']}")
            durations_used = {
                "prx": dinfo["single_gate_time_ns"],
                "cz": dinfo["cz_gate_time_ns"],
                "measure": dinfo["measure_time_ns"],
            }
            for w in dinfo["health_warnings"]:
                print(f"{tag} HEALTH: {w}")
        elif noise_source == "iqm-fake-backend":
            print(f"{tag} IQM fake backend: {prep.info.get('fake_backend')}")

        if prep.relaxation_active:
            # Gate-time relaxation is resident in the noise model; idle/delay
            # relaxation runs as a NoiseModel custom pass inside Aer at assemble
            # time. Statevector samples all of it per shot (scales to large n).
            print(f"{tag} duration-aware relaxation: gate relaxation resident "
                  f"in noise model; idle/delay relaxation via custom pass")

        print(f"{tag} running...")
        job = simulator.run(run_circuits, shots=num_shots, memory=True)
        result = job.result()

        autocorrelators = np.zeros(num_max_kicks)
        counts_per_kick = []
        for i in range(num_max_kicks):
            counts_i = result.get_counts(i)
            autocorrelators[i] = get_autocorrelation(counts_i, init_bit_array, num_qubits)
            counts_per_kick.append({str(k): int(v) for k, v in counts_i.items()})

        elapsed = time.time() - t0
        print(f"{tag} runtime: {elapsed:.2f} s")

        stem = f"instance_{instance_id:02d}"
        dat_path = os.path.join(output_dir, f"{stem}_autocorr.dat")
        json_path = os.path.join(output_dir, f"{stem}_full.json")

        with open(dat_path, "w", encoding="utf-8") as f:
            f.write("# kick   autocorrelator\n")
            for n in range(num_max_kicks):
                f.write(f"{n:4d} {autocorrelators[n]:10.4f}\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump({
                "instance_id": instance_id, "noise_source": noise_source,
                "calibration": calibration_path, "t2_mode": t2_mode,
                "durations_ns": durations_used,
                "num_qubits": num_qubits, "num_shots": num_shots,
                "num_max_kicks": num_max_kicks, "epsilon": epsilon, "h_x": h_x,
                "initial_state": initial_state, "init_bit_array": init_bit_array,
                "Jz_angles": Jz_angles.tolist(), "hz_angles": hz_angles.tolist(),
                "autocorrelators": autocorrelators.tolist(),
                "counts_per_kick": counts_per_kick, "elapsed_seconds": elapsed,
                # Logical-vs-native circuit complexity (device-calibrated only;
                # null for noiseless / iqm-fake-backend, which do not native-
                # compile via the device path). Computed in the prepare() seam.
                "circuit_metrics": prep.info.get("circuit_metrics") or None,
                # IQM fake backend class name when applicable, else null.
                "fake_backend": prep.info.get("fake_backend"),
            }, f, indent=2)

        print(f"{tag} wrote {dat_path}")
        print(f"{tag} wrote {json_path}")
        return (instance_id, True, elapsed, None)

    except Exception as e:
        traceback.print_exc(file=log_fh)
        return (instance_id, False, 0.0, repr(e))
    finally:
        log_fh.flush()
        log_fh.close()
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--noise-source", choices=NOISE_SOURCES, required=True)
    parser.add_argument("--calibration", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-instances", type=int, default=40)
    parser.add_argument("--num-qubits", type=int, default=10)
    parser.add_argument("--num-shots", type=int, default=1000)
    parser.add_argument("--num-max-kicks", type=int, default=60)
    parser.add_argument("--epsilon", type=float, default=0.03)
    parser.add_argument("--initial-state", type=int, default=3)
    parser.add_argument("--t2-mode", choices=["ramsey", "echo"], default="ramsey")
    parser.add_argument(
        "--noise", default="all",
        help="device-calibrated noise channels (default: all). One of "
             "all|none, or a comma-separated subset of "
             "1q,2q,measurement,thermal_relaxation_error "
             "(e.g. --noise=1q,2q or --noise=thermal_relaxation_error). "
             "Ignored for noiseless / iqm-fake-backend.")
    parser.add_argument("--prx-time-ns", type=float, default=None)
    parser.add_argument("--cz-time-ns", type=float, default=None)
    parser.add_argument("--measure-time-ns", type=float, default=None)
    parser.add_argument("--iqm-device", choices=["aphrodite", "apollo"], default="aphrodite")
    parser.add_argument("--single-instance-id", type=int, default=None)
    args = parser.parse_args()

    needs_cal = args.noise_source in ("device-calibrated",)
    if needs_cal:
        if not args.calibration:
            sys.exit(f"ERROR: --calibration required for {args.noise_source}")
        if not os.path.isfile(args.calibration):
            sys.exit(f"ERROR: calibration file not found: {args.calibration}")
    if args.noise_source == "device-calibrated" and not HAVE_DEVICE_NOISE:
        sys.exit("ERROR: device_noise not importable. Set PROJECT_DIR/PYTHONPATH.")

    # Parse the --noise channel selection (device-calibrated only).
    noise_spec = None
    if args.noise_source == "device-calibrated":
        if parse_noise_spec is None:
            sys.exit("ERROR: noise_spec not importable. Set PROJECT_DIR/PYTHONPATH.")
        try:
            noise_spec = parse_noise_spec(args.noise)
        except ValueError as e:
            sys.exit(f"ERROR: bad --noise value '{args.noise}': {e}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.environ.setdefault("QISKIT_IN_PARALLEL", "TRUE")
    durations = (args.prx_time_ns, args.cz_time_ns, args.measure_time_ns)

    print("=" * 64)
    print(" Floquet driver")
    print("=" * 64)
    print(f"noise_source  : {args.noise_source}")
    print(f"calibration   : {args.calibration}")
    print(f"output_dir    : {args.output_dir}")
    print(f"num_instances : {args.num_instances}")
    print(f"num_qubits    : {args.num_qubits}")
    print(f"num_shots     : {args.num_shots}")
    print(f"num_max_kicks : {args.num_max_kicks}")
    print(f"epsilon       : {args.epsilon}")
    print(f"initial_state : {args.initial_state}")
    if args.noise_source == "device-calibrated":
        print(f"t2_mode       : {args.t2_mode}")
        print(f"noise         : {noise_spec.describe()}")
    if args.noise_source == "iqm-fake-backend":
        print(f"iqm_device    : {args.iqm_device}")
    print("=" * 64)
    sys.stdout.flush()

    worker_args = [
        (i, args.noise_source, args.calibration, args.output_dir, args.num_qubits,
         args.num_shots, args.num_max_kicks, args.epsilon, args.initial_state,
         args.t2_mode, durations, args.iqm_device, noise_spec)
        for i in range(args.num_instances)]

    if args.single_instance_id is not None:
        if not 0 <= args.single_instance_id < args.num_instances:
            sys.exit("ERROR: --single-instance-id out of range")
        print(f"DEBUG: running ONLY instance {args.single_instance_id} (no Pool)")
        sys.stdout.flush()
        r = run_one_instance(worker_args[args.single_instance_id])
        print(f"\nresult: {r}")
        log_path = os.path.join(args.output_dir, f"instance_{r[0]:02d}.log")
        if os.path.exists(log_path):
            print(f"\n----- {log_path} -----")
            with open(log_path, encoding="utf-8") as f:
                sys.stdout.write(f.read())
            print(f"----- end {log_path} -----")
        sys.exit(0 if r[1] else 1)

    print(f"Launching {args.num_instances} workers via multiprocessing.Pool(forkserver)...")
    sys.stdout.flush()
    t0 = time.time()
    ctx = mp.get_context("forkserver")
    with ctx.Pool(args.num_instances) as pool:
        results = pool.map(run_one_instance, worker_args)
    wall = time.time() - t0

    n_ok = sum(1 for r in results if r[1])
    print()
    print("-" * 64)
    print(f"Pool wall time : {wall:.2f} s")
    print(f"Completed      : {n_ok}/{len(results)}")
    print("-" * 64)
    for r in results:
        iid, ok, t_run, err = r
        if ok:
            print(f"  instance {iid:02d}  OK    runtime={t_run:6.2f}s")
        else:
            print(f"  instance {iid:02d}  FAIL  {err}")
    print("-" * 64)
    print(f"Per-instance logs: {args.output_dir}/instance_*.log")
    print(f"Per-instance dat : {args.output_dir}/instance_*_autocorr.dat")
    print(f"Per-instance json: {args.output_dir}/instance_*_full.json")
    sys.exit(0 if n_ok == len(results) else 1)


if __name__ == "__main__":
    main()
