#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""
Floquet driver -- N gate instances in parallel via multiprocessing.

ONE Python process, ONE container, ONE srun; parallelism via
multiprocessing.Pool(forkserver). qiskit / qiskit-aer / HPCQC noise builders
imported at module level so workers inherit them.

Noise sources (--noise-source):
  noiseless        : AerSimulator(), no noise, circuit as-written.
  logical-gates    : your logical circuit as-written (rx/rz/rzz, no native
                     transpilation or routing), with HPCQC's existing
                     calibration noise model (build_noise_model). Fast.
  device-calibrated: transpile to device native gates (PRX->r, CZ->cz),
                     route onto the device coupling map, SCHEDULE
                     (scheduling done inside transpile) so idle periods are
                     explicit delays,
                     then simulate with a static NoiseModel of depolarizing
                     CONTROL error (gates) + readout, plus a duration-aware
                     RelaxationNoisePass applying ALL T1/T2 decoherence to
                     every instruction (gates AND idle delays) by its real
                     scheduled duration. Most faithful. (device_noise.py)
  iqm-fake-backend : run on a genuine IQM IQMFake* backend as a static
                     representative reference. Dependency call; not HPCQC.

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
    from lumi_hpc_qc.backends.noise_model import build_noise_model
    HAVE_LOGICAL_NOISE = True
except ImportError:
    _proj = os.environ.get("PROJECT_DIR") or os.environ.get("SINGULARITYENV_PROJECT_DIR")
    if _proj:
        sys.path.insert(0, os.path.join(_proj, "src"))
    try:
        from lumi_hpc_qc.backends.noise_model import build_noise_model
        HAVE_LOGICAL_NOISE = True
    except ImportError:
        build_noise_model = None
        HAVE_LOGICAL_NOISE = False

try:
    from lumi_hpc_qc.backends.device_noise import (
        build_control_readout_noise_model,
        build_relaxation_pass,
    )
    HAVE_DEVICE_NOISE = True
except ImportError:
    build_control_readout_noise_model = None
    build_relaxation_pass = None
    HAVE_DEVICE_NOISE = False

NOISE_SOURCES = ["noiseless", "logical-gates", "device-calibrated", "iqm-fake-backend"]


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


def _prepare_device_circuits(circuits, num_qubits, calibration_path, t2_mode,
                             durations, tag):
    """Turn logical circuits into scheduled, native-gate circuits ready to run.

    This does the steps that make a circuit resemble what the real Q50 would
    execute, and returns everything the caller needs to run them:

      1. Translate every gate into the device's native gate set (the only
         gates the hardware physically has): PRX -> Qiskit "r", CZ -> "cz".
      2. Route the circuit onto the device's actual qubit connectivity,
         inserting SWAPs where two qubits that need to interact are not
         physically adjacent.
      3. Schedule it: lay every gate out on a timeline using the gate
         durations, and fill each qubit's idle stretches with explicit
         "delay" steps. Crucially, scheduling also stamps a concrete duration
         onto EVERY instruction (each gate and each delay), not just the
         delays.
      4. Build the matching noise: the static gate/readout noise model, plus
         the time-based relaxation pass the caller applies to the scheduled
         circuit.

    Steps 1-3 are done in a single transpile() call with
    scheduling_method='alap'. This matters: the relaxation pass in step 4
    reads each instruction's OWN .duration attribute to decide how much it
    decoheres. If scheduling is done as a separate pass afterwards, the gates
    end up without a .duration attached, and the relaxation pass silently
    skips every gate (applying decoherence only to the delays) -- which
    quietly removes most of the physics. Scheduling inside transpile attaches
    durations to all instructions, so the relaxation pass sees them.

    On units: time is measured in integer "ticks" (Qiskit calls one tick
    "dt"); we set one tick = 1 nanosecond (dt = 1e-9 s), so a gate we describe
    as lasting "60" is 60 ns. The relaxation pass is given the same dt to
    convert each instruction's tick-count back into seconds.

    Returns:
        (scheduled_circuits, simulator, relaxation_pass, info)
    """
    from qiskit.transpiler import (InstructionDurations, CouplingMap, Target,
                                   PassManager)

    sg_ns, cz_ns, me_ns = durations
    nm, coupling_map, info = build_control_readout_noise_model(
        calibration_path, num_qubits=num_qubits, t2_mode=t2_mode,
        single_gate_time_ns=sg_ns, cz_gate_time_ns=cz_ns, measure_time_ns=me_ns)

    # One tick = 1 ns, so a duration value of N means N nanoseconds.
    dt_s = 1e-9
    sg = int(round(info["single_gate_time_ns"]))   # PRX:    ~20 ns
    cz = int(round(info["cz_gate_time_ns"]))        # CZ:     ~60 ns
    me = int(round(info["measure_time_ns"]))        # readout ~1576 ns

    # Native gate set the hardware supports (plus measure/id helpers).
    # rz is a virtual frame change on real hardware, so it takes zero time.
    basis = ["r", "rz", "sx", "x", "cz", "id", "measure"]
    instr_durations = InstructionDurations(
        [
            ("r", None, sg), ("rz", None, 0), ("sx", None, sg),
            ("x", None, sg), ("id", None, sg), ("cz", None, cz),
            ("measure", None, me), ("reset", None, me),
        ],
        dt=dt_s,
    )

    cmap = coupling_map if isinstance(coupling_map, CouplingMap) else (
        CouplingMap(coupling_map) if coupling_map else None)

    # Qiskit 2.3's transpile() removed the loose instruction_durations kwarg;
    # durations must be carried on a Target. We build a Target from the native
    # basis, coupling map, and durations, then pass it to transpile with
    # scheduling_method='alap'. transpile internally builds a preset pass
    # manager with this target and schedules with it -- doing layout, routing,
    # native translation, and ALAP scheduling in one validated pipeline.
    target = Target.from_configuration(
        basis_gates=basis,
        num_qubits=num_qubits,
        coupling_map=cmap,
        instruction_durations=instr_durations,
        dt=dt_s,
    )
    scheduled = transpile(
        circuits,
        target=target,
        scheduling_method="alap",
        optimization_level=3,
        num_processes=1,
    )

    # Step 4: the time-based decoherence pass. CRUCIAL: we pass the SAME target
    # to the pass. Aer's RelaxationNoisePass looks up GATE durations from the
    # target (by name+qubits), and reads DELAY durations off the scheduled
    # circuit. Without the target, it would try to read gate .duration off the
    # circuit, not find it, and silently skip every gate (the cause of the
    # "Instruction duration not found" warnings seen earlier). Passing the
    # target is exactly how Aer's own NoiseModel.from_backend builds its
    # relaxation pass.
    relax_pass, _, _ = build_relaxation_pass(
        calibration_path, num_qubits=num_qubits, t2_mode=t2_mode,
        dt_seconds=dt_s, target=target)

    # NOTE on why we apply the pass MANUALLY here rather than attaching it to
    # the noise model: Aer only runs a noise model's custom passes on circuits
    # that already contain a Delay instruction (see aerbackend.py: it filters
    # to circuits where `Delay in optypes`). Our Floquet circuit is uniform --
    # every qubit is driven every layer and all qubits are measured together --
    # so ALAP scheduling may insert few or no delays, and Aer would then SKIP
    # the relaxation pass entirely (applying neither gate nor idle relaxation).
    # Applying the pass ourselves guarantees it runs regardless. The pass
    # inserts thermal-relaxation channels as instructions in the circuit; Aer
    # recognises and applies those channel instructions at run time (it pulls
    # them into the noise model internally), so we can run the result directly.
    relax_pm = PassManager([relax_pass])
    scheduled = relax_pm.run(scheduled)

    # Keep each worker single-threaded (40 workers share the node; we don't
    # want each one spawning its own Aer thread pool).
    aer_threading = dict(max_parallel_threads=1, max_parallel_experiments=1,
                         max_parallel_shots=1)
    simulator = AerSimulator(noise_model=nm, **aer_threading)
    # The relaxation channels are now baked into `scheduled` (applied above),
    # so the worker runs these circuits directly. We still return relax_pass
    # so the caller can log that relaxation was applied.
    return scheduled, simulator, relax_pass, info


def run_one_instance(args_tuple):
    (instance_id, noise_source, calibration_path, output_dir, num_qubits,
     num_shots, num_max_kicks, epsilon, initial_state, t2_mode, durations,
     iqm_device) = args_tuple

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
        aer_threading = dict(max_parallel_threads=1, max_parallel_experiments=1,
                             max_parallel_shots=1)

        print(f"{tag} noise-source: {noise_source}")
        print(f"{tag} init_bit_array = {init_bit_array}")
        t0 = time.time()

        Jz_angles = np.random.uniform(-1.5 * np.pi, -0.5 * np.pi, num_qubits)
        hz_angles = np.random.uniform(-np.pi, np.pi, num_qubits)
        circuits = [build_circuit(n, hz_angles, Jz_angles, init_bit_array,
                                  num_qubits, h_x) for n in range(num_max_kicks)]

        relax_pass = None
        # Actual durations used (ns). For device-calibrated this is filled from
        # the resolved values (file or CLI override); for other sources the
        # CLI override tuple (often all None) is recorded as-is.
        durations_used = {"prx": durations[0], "cz": durations[1],
                          "measure": durations[2]}

        if noise_source == "noiseless":
            simulator = AerSimulator(**aer_threading)
            print(f"{tag} transpiling {len(circuits)} circuits (serial)...")
            run_circuits = transpile(circuits, simulator,
                                     optimization_level=3, num_processes=1)

        elif noise_source == "logical-gates":
            if not HAVE_LOGICAL_NOISE:
                raise RuntimeError("build_noise_model not importable")
            nm, coupling_map = build_noise_model(
                calibration_path, num_qubits=num_qubits, noise_channels=None)
            simulator = AerSimulator(noise_model=nm, coupling_map=coupling_map,
                                     **aer_threading)
            print(f"{tag} WARNING: logical-gates simulates abstract gates (rzz) "
                  f"without native transpilation; 2q noise may under-attach.")
            print(f"{tag} transpiling {len(circuits)} circuits (serial)...")
            run_circuits = transpile(circuits, simulator,
                                     optimization_level=3, num_processes=1)

        elif noise_source == "device-calibrated":
            if not HAVE_DEVICE_NOISE:
                raise RuntimeError("device_noise not importable")
            print(f"{tag} transpiling to native gates + scheduling...")
            run_circuits, simulator, relax_pass, dinfo = _prepare_device_circuits(
                circuits, num_qubits, calibration_path, t2_mode, durations, tag)
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
            from iqm.qiskit_iqm import IQMFakeAphrodite
            try:
                from iqm.qiskit_iqm import IQMFakeApollo
            except Exception:
                IQMFakeApollo = None
            fake_map = {"aphrodite": IQMFakeAphrodite, "apollo": IQMFakeApollo}
            fb_cls = fake_map.get(iqm_device) or IQMFakeAphrodite
            fb = fb_cls()
            print(f"{tag} IQM fake backend: {fb_cls.__name__}")
            print(f"{tag} transpiling for fake backend...")
            run_circuits = transpile(circuits, backend=fb,
                                     optimization_level=3, num_processes=1)
            simulator = fb
        else:
            raise ValueError(f"unknown noise-source {noise_source!r}")

        if relax_pass is not None:
            # The duration-aware relaxation was already applied to the
            # scheduled circuits inside _prepare_device_circuits (the channels
            # are baked into run_circuits). Aer applies those channel
            # instructions at run time. Nothing to do here but note it.
            print(f"{tag} duration-aware relaxation: applied to scheduled "
                  f"circuits (gate + idle decoherence)")

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
    parser.add_argument("--prx-time-ns", type=float, default=None)
    parser.add_argument("--cz-time-ns", type=float, default=None)
    parser.add_argument("--measure-time-ns", type=float, default=None)
    parser.add_argument("--iqm-device", choices=["aphrodite", "apollo"], default="aphrodite")
    parser.add_argument("--single-instance-id", type=int, default=None)
    args = parser.parse_args()

    needs_cal = args.noise_source in ("logical-gates", "device-calibrated")
    if needs_cal:
        if not args.calibration:
            sys.exit(f"ERROR: --calibration required for {args.noise_source}")
        if not os.path.isfile(args.calibration):
            sys.exit(f"ERROR: calibration file not found: {args.calibration}")
    if args.noise_source == "device-calibrated" and not HAVE_DEVICE_NOISE:
        sys.exit("ERROR: device_noise not importable. Set PROJECT_DIR/PYTHONPATH.")

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
    if args.noise_source == "iqm-fake-backend":
        print(f"iqm_device    : {args.iqm_device}")
    print("=" * 64)
    sys.stdout.flush()

    worker_args = [
        (i, args.noise_source, args.calibration, args.output_dir, args.num_qubits,
         args.num_shots, args.num_max_kicks, args.epsilon, args.initial_state,
         args.t2_mode, durations, args.iqm_device)
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
