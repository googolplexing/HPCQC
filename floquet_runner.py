#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""
Floquet driver -- 10 (or N) gate instances in parallel via multiprocessing.

Architecture: ONE Python process, ONE container instance, started by ONE
``srun`` -- matches the proven HPCQC standard-partition pattern
(``tests/e2_cpu_stress_test.py``, ``tests/slurm_e2_stress.sh``).
Parallelism is provided by ``multiprocessing.Pool`` with the ``forkserver``
context. qiskit / qiskit-aer / the HPCQC noise builder are imported once
at MODULE LEVEL (before the forkserver starts) so workers inherit them
and don't each re-trigger the qiskit-aer C++ extension load.

Inside each worker, ``transpile(..., num_processes=1)`` is required:
Pool workers are daemonic, and daemonic processes can't spawn child
processes -- which is what qiskit's ``parallel_map`` would otherwise try
to do when transpiling a list of circuits with optimization_level=3.
Setting QISKIT_IN_PARALLEL=TRUE in the env (the SLURM script does this)
is a belt-and-braces against any other nested parallel_map inside
qiskit/qiskit-aer.

The per-circuit physics (build_circuit, apply_one_floquet_period,
get_autocorrelation, FFT inputs) is byte-identical to the original
driver. Only the outer "for n in range(num_gate_instances)" became a
worker pool.

Usage:
  python3 floquet_runner.py --backend noiseless \\
                            --output-dir results/floquet_noiseless \\
                            --num-instances 10

  python3 floquet_runner.py --backend q50-noise \\
                            --calibration examples/q50_calibration_<...>.json \\
                            --output-dir results/floquet_q50noise \\
                            --num-instances 10

  # debug a single instance (skips the pool):
  python3 floquet_runner.py --backend noiseless \\
                            --output-dir /tmp/dbg --single-instance-id 0
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

# -- Import qiskit/aer at MODULE LEVEL so the forkserver process loads them
# once and child workers inherit. Matches tests/e2_cpu_stress_test.py.
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit_aer import AerSimulator

# -- Optional: HPCQC noise builder (only needed for --backend=q50-noise).
try:
    from lumi_hpc_qc.backends.noise_model import build_noise_model
    HAVE_NOISE_BUILDER = True
except ImportError:
    _proj = os.environ.get("PROJECT_DIR") or os.environ.get("SINGULARITYENV_PROJECT_DIR")
    if _proj:
        sys.path.insert(0, os.path.join(_proj, "src"))
    try:
        from lumi_hpc_qc.backends.noise_model import build_noise_model
        HAVE_NOISE_BUILDER = True
    except ImportError:
        build_noise_model = None  # type: ignore
        HAVE_NOISE_BUILDER = False


# ----------------- physics helpers (unchanged) -----------------

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


def build_circuit(num_kicks, hz_angles, Jzz_angles, init_bit_array,
                  num_qubits, h_x):
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


# ----------------- worker: one gate instance -----------------

def run_one_instance(args_tuple):
    """Worker -- runs ONE gate instance, writes its own log/dat/json,
    returns (instance_id, ok, runtime_s, err_msg|None)."""
    (instance_id, backend, calibration_path, output_dir,
     num_qubits, num_shots, num_max_kicks, epsilon, initial_state) = args_tuple

    tag = f"[instance {instance_id:02d}]"
    log_path = os.path.join(output_dir, f"instance_{instance_id:02d}.log")

    # Belt-and-braces: tell qiskit's parallel_map to stay serial inside
    # this daemonic worker (the env var is what SLURM script also sets;
    # we re-export here in case the runner is invoked outside SLURM).
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

        # Force each Pool worker to run Aer strictly single-threaded, so
        # 40 workers map to ~40 cores instead of 40 * (Aer threads)
        # oversubscribing the 128-core node. These are Aer run-options;
        # OMP_NUM_THREADS etc. (set in the SLURM script) cap NumPy/BLAS,
        # but Aer's own pool is controlled here.
        aer_threading = dict(
            max_parallel_threads=1,
            max_parallel_experiments=1,
            max_parallel_shots=1,
        )

        if backend == "noiseless":
            print(f"{tag} backend: noiseless AerSimulator()")
            simulator = AerSimulator(**aer_threading)
        else:
            print(f"{tag} backend: AerSimulator + Q50 calibration noise model "
                  f"(all 5 channels)")
            print(f"{tag} calibration: {calibration_path}")
            if not HAVE_NOISE_BUILDER:
                raise RuntimeError(
                    "lumi_hpc_qc.backends.noise_model not importable; "
                    "ensure HPCQC src/ is on PYTHONPATH (set PROJECT_DIR)."
                )
            noise_model, coupling_map = build_noise_model(
                calibration_path,
                num_qubits=num_qubits,
                noise_channels=None,   # None => all 5 channels active
            )
            simulator = AerSimulator(noise_model=noise_model,
                                     coupling_map=coupling_map,
                                     **aer_threading)

        print(f"{tag} init_bit_array = {init_bit_array}")

        t0 = time.time()

        Jz_angles = np.random.uniform(-1.5 * np.pi, -0.5 * np.pi, num_qubits)
        hz_angles = np.random.uniform(-np.pi,        np.pi,       num_qubits)

        circuits = [
            build_circuit(n, hz_angles, Jz_angles, init_bit_array,
                          num_qubits, h_x)
            for n in range(num_max_kicks)
        ]

        # num_processes=1: critical inside Pool workers -- without it, qiskit's
        # parallel_map tries to spawn child processes which is forbidden in
        # daemonic Pool workers (AssertionError "daemonic processes are not
        # allowed to have children").
        print(f"{tag} transpiling {len(circuits)} circuits (serial)...")
        compiled = transpile(circuits, simulator,
                             optimization_level=3, num_processes=1)

        print(f"{tag} running...")
        job = simulator.run(compiled, shots=num_shots, memory=True)
        result = job.result()

        autocorrelators = np.zeros(num_max_kicks)
        counts_per_kick = []
        for i in range(num_max_kicks):
            counts_i = result.get_counts(i)
            autocorrelators[i] = get_autocorrelation(counts_i,
                                                     init_bit_array,
                                                     num_qubits)
            counts_per_kick.append({str(k): int(v) for k, v in counts_i.items()})

        elapsed = time.time() - t0
        print(f"{tag} runtime: {elapsed:.2f} s")

        stem      = f"instance_{instance_id:02d}"
        dat_path  = os.path.join(output_dir, f"{stem}_autocorr.dat")
        json_path = os.path.join(output_dir, f"{stem}_full.json")

        with open(dat_path, "w", encoding="utf-8") as f:
            f.write("# kick   autocorrelator\n")
            for n in range(num_max_kicks):
                f.write(f"{n:4d} {autocorrelators[n]:10.4f}\n")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "instance_id":     instance_id,
                    "backend":         backend,
                    "calibration":     calibration_path,
                    "num_qubits":      num_qubits,
                    "num_shots":       num_shots,
                    "num_max_kicks":   num_max_kicks,
                    "epsilon":         epsilon,
                    "h_x":             h_x,
                    "initial_state":   initial_state,
                    "init_bit_array":  init_bit_array,
                    "Jz_angles":       Jz_angles.tolist(),
                    "hz_angles":       hz_angles.tolist(),
                    "autocorrelators": autocorrelators.tolist(),
                    "counts_per_kick": counts_per_kick,
                    "elapsed_seconds": elapsed,
                },
                f, indent=2,
            )

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


# ----------------- main -----------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--backend",        choices=["noiseless", "q50-noise"],
                        required=True)
    parser.add_argument("--calibration",    default=None,
                        help="HPCQC-format Q50 calibration JSON. "
                             "Required when --backend=q50-noise.")
    parser.add_argument("--output-dir",     required=True)
    parser.add_argument("--num-instances",  type=int,   default=10)
    parser.add_argument("--num-qubits",     type=int,   default=10)
    parser.add_argument("--num-shots",      type=int,   default=100)
    parser.add_argument("--num-max-kicks",  type=int,   default=40)
    parser.add_argument("--epsilon",        type=float, default=0.03)
    parser.add_argument("--initial-state",  type=int,   default=3,
                        help="1: random, 2: Neel, 3: polarized (default)")
    parser.add_argument("--single-instance-id", type=int, default=None,
                        help="(debug) run only this instance and exit; skip Pool")
    args = parser.parse_args()

    if args.backend == "q50-noise":
        if not args.calibration:
            sys.exit("ERROR: --calibration required when --backend=q50-noise")
        if not os.path.isfile(args.calibration):
            sys.exit(f"ERROR: calibration file not found: {args.calibration}")
        if not HAVE_NOISE_BUILDER:
            sys.exit("ERROR: lumi_hpc_qc.backends.noise_model not importable. "
                     "Set PROJECT_DIR or PYTHONPATH so HPCQC src/ is reachable.")

    os.makedirs(args.output_dir, exist_ok=True)

    # Make qiskit's parallel_map stay serial in this process and any
    # children it forks. Also re-asserted inside the worker.
    os.environ.setdefault("QISKIT_IN_PARALLEL", "TRUE")

    print("=" * 64)
    print(" Floquet driver")
    print("=" * 64)
    print(f"backend       : {args.backend}")
    print(f"calibration   : {args.calibration}")
    print(f"output_dir    : {args.output_dir}")
    print(f"num_instances : {args.num_instances}")
    print(f"num_qubits    : {args.num_qubits}")
    print(f"num_shots     : {args.num_shots}")
    print(f"num_max_kicks : {args.num_max_kicks}")
    print(f"epsilon       : {args.epsilon}")
    print(f"initial_state : {args.initial_state}")
    print("=" * 64)
    sys.stdout.flush()

    worker_args = [
        (i, args.backend, args.calibration, args.output_dir,
         args.num_qubits, args.num_shots, args.num_max_kicks,
         args.epsilon, args.initial_state)
        for i in range(args.num_instances)
    ]

    # -- debug path: skip the Pool, run one instance in-process --
    if args.single_instance_id is not None:
        if not 0 <= args.single_instance_id < args.num_instances:
            sys.exit(f"ERROR: --single-instance-id={args.single_instance_id} "
                     f"out of range [0, {args.num_instances})")
        print(f"DEBUG: running ONLY instance {args.single_instance_id} "
              f"(no Pool)")
        sys.stdout.flush()
        r = run_one_instance(worker_args[args.single_instance_id])
        print(f"\nresult: {r}")
        log_path = os.path.join(args.output_dir,
                                f"instance_{r[0]:02d}.log")
        if os.path.exists(log_path):
            print(f"\n----- {log_path} -----")
            with open(log_path, encoding="utf-8") as f:
                sys.stdout.write(f.read())
            print(f"----- end {log_path} -----")
        sys.exit(0 if r[1] else 1)

    # -- parallel path: forkserver Pool --
    print(f"Launching {args.num_instances} workers via "
          f"multiprocessing.Pool(forkserver)...")
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
