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
    from lumi_hpc_qc.backends.device_noise import (
        build_control_readout_noise_model,
        build_relaxation_pass,
    )
    HAVE_DEVICE_NOISE = True
except ImportError:
    # Not pip-installed: make the in-repo package importable from a checkout
    # (PROJECT_DIR/src), then retry. This sys.path bootstrap previously lived
    # in the now-removed logical-gates (build_noise_model) import block.
    _proj = os.environ.get("PROJECT_DIR") or os.environ.get("SINGULARITYENV_PROJECT_DIR")
    if _proj:
        sys.path.insert(0, os.path.join(_proj, "src"))
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

# parse_noise_spec has no qiskit dependency; import it independently so the
# --noise flag still validates even if the qiskit-dependent builders are absent.
try:
    from lumi_hpc_qc.backends.noise_spec import parse_noise_spec
except ImportError:
    parse_noise_spec = None

NOISE_SOURCES = ["noiseless", "device-calibrated", "iqm-fake-backend"]


def resolve_instance_seed(master_seed, instance_id):
    """Return a reproducible, per-instance integer seed (or None for fresh
    entropy), using numpy SeedSequence so instances are statistically
    independent yet deterministic from one master seed.

    master_seed semantics:
      None or "random" -> fresh OS entropy each run (NOT reproducible); returns
                          None, which both global RNG seeding and Aer treat as
                          "draw from entropy".
      <int>            -> reproducible: instance k always gets the same seed,
                          and different master_seeds give independent ensembles.

    The single integer returned is used to seed BOTH the global numpy/`random`
    state (for circuit-construction draws) and Aer's seed_simulator (for shot +
    noise-trajectory sampling), so the entire instance is reproducible together.
    """
    if master_seed is None or master_seed == "random":
        return None
    # SeedSequence.spawn gives independent child sequences; take a 32-bit int
    # from child[instance_id]. 32-bit keeps it in range for Aer's seed_simulator.
    child = np.random.SeedSequence(int(master_seed)).spawn(instance_id + 1)[instance_id]
    return int(child.generate_state(1, dtype=np.uint32)[0])


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
                             durations, tag, spec=None):
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
    from qiskit.transpiler import InstructionDurations, CouplingMap, Target

    sg_ns, cz_ns, me_ns = durations
    nm, coupling_map, info = build_control_readout_noise_model(
        calibration_path, num_qubits=num_qubits, t2_mode=t2_mode,
        single_gate_time_ns=sg_ns, cz_gate_time_ns=cz_ns, measure_time_ns=me_ns,
        spec=spec)

    # One tick = 1 ns, so a duration value of N means N nanoseconds.
    dt_s = 1e-9
    sg = int(round(info["single_gate_time_ns"]))   # PRX:    ~20 ns
    cz = int(round(info["cz_gate_time_ns"]))        # CZ:     ~60 ns
    me = int(round(info["measure_time_ns"]))        # readout ~1576 ns

    # Native gate set the hardware supports (plus measure/id helpers).
    # rz is a virtual frame change on real hardware and takes zero time.
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

    # Idle/delay relaxation. Gate-time relaxation is already RESIDENT in `nm`
    # (added by build_control_readout_noise_model); this pass adds the
    # variable-duration part -- the idle "delay" steps the scheduler inserts.
    # We register it as a NoiseModel custom noise pass, exactly how Aer's own
    # NoiseModel.from_backend attaches its delay-relaxation pass: Aer runs it at
    # assemble time on circuits containing a Delay and skips those with none.
    # Skipped entirely when the thermal channel is disabled (e.g. --noise=1q,2q).
    relax_pass = None
    thermal_on = (spec is None) or bool(spec.thermal_relaxation)
    if thermal_on:
        relax_pass, _, _ = build_relaxation_pass(
            calibration_path, num_qubits=num_qubits, t2_mode=t2_mode,
            dt_seconds=dt_s, target=target)
        nm._custom_noise_passes.append(relax_pass)

    # Keep each worker single-threaded (40 workers share the node; we don't
    # want each one spawning its own Aer thread pool).
    aer_threading = dict(max_parallel_threads=1, max_parallel_experiments=1,
                         max_parallel_shots=1)
    # Pin method="statevector" rather than leaving it "automatic". The thermal
    # channel is a genuine non-unitary Kraus map for any qubit with T2 > T1.
    # Under statevector, Aer samples such noise via a kraus path that reads a
    # PRECOMPUTED canonical Kraus (circuit_executor.hpp: branch on
    # opset.contains(kraus)). That precompute (NoiseModel::enable_kraus_method)
    # runs DETERMINISTICALLY in the forced-method path when method==statevector
    # and the model opset contains kraus (aer_controller.hpp:778). In "automatic"
    # mode the same precompute is gated on a per-circuit method decision that can
    # be skipped, leaving the kraus path with an empty Kraus -> "QuantumError:
    # Kraus is empty". Forcing statevector closes that gating gap for EVERY
    # --noise combination (including thermal-only) and is the scalable path
    # (statevector + per-shot sampling, not O(4^n) density_matrix).
    simulator = AerSimulator(method="statevector", noise_model=nm, **aer_threading)
    # `scheduled` carries the native + routed + ALAP-scheduled circuits with
    # explicit Delays; Aer runs the delay-relaxation pass at assemble time. We
    # return relax_pass so the caller can log whether relaxation is configured.
    return scheduled, simulator, relax_pass, info


def run_one_instance(args_tuple):
    (instance_id, noise_source, calibration_path, output_dir, num_qubits,
     num_shots, num_max_kicks, epsilon, initial_state, t2_mode, durations,
     iqm_device, noise_spec, master_seed) = args_tuple

    tag = f"[instance {instance_id:02d}]"
    log_path = os.path.join(output_dir, f"instance_{instance_id:02d}.log")
    os.environ["QISKIT_IN_PARALLEL"] = "TRUE"

    log_fh = open(log_path, "w", buffering=1, encoding="utf-8")
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    sys.stdout = log_fh
    sys.stderr = log_fh

    try:
        # Resolve ONE per-instance seed and apply it to the global numpy/random
        # state immediately before any circuit-construction draws, so those
        # draws are reproducible. The SAME seed sets seed_simulator below, so
        # the whole instance (circuit + sampling) reproduces together. Nothing
        # may consume the global RNG between here and the draws.
        inst_seed = resolve_instance_seed(master_seed, instance_id)
        if inst_seed is not None:
            random.seed(inst_seed)
            np.random.seed(inst_seed)
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

        elif noise_source == "device-calibrated":
            if not HAVE_DEVICE_NOISE:
                raise RuntimeError("device_noise not importable")
            print(f"{tag} transpiling to native gates + scheduling...")
            if noise_spec is not None:
                print(f"{tag} noise channels: {noise_spec.describe()}")
            run_circuits, simulator, relax_pass, dinfo = _prepare_device_circuits(
                circuits, num_qubits, calibration_path, t2_mode, durations, tag,
                spec=noise_spec)
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
            # Gate-time relaxation is resident in the noise model; idle/delay
            # relaxation runs as a NoiseModel custom pass inside Aer at assemble
            # time. Statevector samples all of it per shot (scales to large n).
            print(f"{tag} duration-aware relaxation: gate relaxation resident "
                  f"in noise model; idle/delay relaxation via custom pass")

        seed_note = "entropy (not reproducible)" if inst_seed is None else inst_seed
        print(f"{tag} running... (seed: {seed_note})")
        job = simulator.run(run_circuits, shots=num_shots, memory=True,
                            seed_simulator=inst_seed)
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
                # Reproducibility provenance: master_seed is the run-wide knob;
                # instance_seed is the resolved per-instance seed actually used
                # for both circuit draws and seed_simulator (null = fresh
                # entropy, run not reproducible).
                "master_seed": master_seed,
                "instance_seed": inst_seed,
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
    parser.add_argument(
        "--master-seed", default="0",
        help="Run-wide RNG seed for reproducibility. An integer (default 0) "
             "makes every instance's circuit draws AND shot/noise sampling "
             "reproducible (instance k derives an independent seed via "
             "numpy SeedSequence); change it for a fresh independent ensemble. "
             "Pass 'random' for fresh OS entropy each run (NOT reproducible).")
    parser.add_argument("--single-instance-id", type=int, default=None)
    args = parser.parse_args()

    # master_seed: int, or the literal string "random" -> None (fresh entropy).
    if str(args.master_seed).lower() == "random":
        master_seed = "random"
    else:
        try:
            master_seed = int(args.master_seed)
        except ValueError:
            sys.exit(f"ERROR: --master-seed must be an integer or 'random', "
                     f"got {args.master_seed!r}")

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
    print(f"master_seed   : {master_seed}"
          + ("  (fresh entropy; NOT reproducible)" if master_seed == "random" else ""))
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
         args.t2_mode, durations, args.iqm_device, noise_spec, master_seed)
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
