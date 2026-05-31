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
    # PLACEMENT-1 / parity reuse: the runner resolves placement and disorder
    # through the SAME code the sweep uses -- no parallel reimplementation.
    #   resolve_placements   : solver seam (manual names -> validated Placement)
    #   resolve_disorder      : source:file load + fail-loud assertions (F2)
    #   extract_connectivity  : circuit 2q-edges from the built circuit
    from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
    from lumi_hpc_qc.sweep.byo_sweep import resolve_disorder
    from lumi_hpc_qc.sweep.circuit_loader import extract_connectivity
    from lumi_hpc_qc.plugins.registry import PluginRegistry
    HAVE_DEVICE_NOISE = True
except ImportError:
    # Not pip-installed: make the in-repo package importable from a checkout
    # (PROJECT_DIR/src), then retry.
    _proj = os.environ.get("PROJECT_DIR") or os.environ.get("SINGULARITYENV_PROJECT_DIR")
    if _proj:
        sys.path.insert(0, os.path.join(_proj, "src"))
    try:
        from lumi_hpc_qc.backends.prepare import prepare_simulation
        from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver
        from lumi_hpc_qc.sweep.byo_sweep import resolve_disorder
        from lumi_hpc_qc.sweep.circuit_loader import extract_connectivity
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        HAVE_DEVICE_NOISE = True
    except ImportError:
        prepare_simulation = None
        GeneralPlacementSolver = None
        resolve_disorder = None
        extract_connectivity = None
        PluginRegistry = None
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

    NOTE: identical to floquet_runner.resolve_instance_seed -- both runners must
    use the same logic for the A/B comparison to be valid.
    """
    if master_seed is None or master_seed == "random":
        return None
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


def run_one_instance(args_tuple):
    (instance_id, noise_source, calibration_path, output_dir, num_qubits,
     num_shots, num_max_kicks, epsilon, initial_state, t2_mode, durations,
     iqm_device, noise_spec, master_seed, disorder_file, physical_qubits) = args_tuple

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

        print(f"{tag} noise-source: {noise_source}")
        t0 = time.time()

        # ── Disorder: file (parity with the sweep, RNG-free) vs. draw (default).
        #    File mode reuses the sweep's resolve_disorder (same load + the F2
        #    fail-loud assertions: num_qubits / initial_state / array lengths /
        #    seed coverage), so both arms consume byte-identical instances from
        #    one banked JSON -- no parallel disorder logic. Draw mode is the
        #    prior legacy-stream behaviour, preserved as the default when no
        #    --disorder-file is given. inst_seed still seeds seed_simulator
        #    below in BOTH modes. ──
        if disorder_file is not None:
            if resolve_disorder is None:
                raise RuntimeError("resolve_disorder not importable")
            resolved, _ = resolve_disorder(
                {"source": "file", "file": disorder_file},
                [instance_id],
                num_qubits=num_qubits,
                configured_initial_state=initial_state,
            )
            inst = resolved[instance_id]
            hz_angles = np.asarray(inst["hz_angles"], dtype=float)
            Jz_angles = np.asarray(inst["Jzz_angles"], dtype=float)
            init_bit_array = list(inst["init_bit_array"])
            print(f"{tag} disorder: file {disorder_file} (instance {instance_id})")
        else:
            init_bit_array = build_init_bit_array(initial_state, num_qubits)
            Jz_angles = np.random.uniform(-1.5 * np.pi, -0.5 * np.pi, num_qubits)
            hz_angles = np.random.uniform(-np.pi, np.pi, num_qubits)
            print(f"{tag} disorder: drawn (master_seed-derived)")
        print(f"{tag} init_bit_array = {init_bit_array}")

        circuits = [build_circuit(n, hz_angles, Jz_angles, init_bit_array,
                                  num_qubits, h_x) for n in range(num_max_kicks)]

        # Actual durations used (ns). For device-calibrated this is overwritten
        # from the resolved values below; for other sources the CLI override
        # tuple (often all None) is recorded as-is.
        durations_used = {"prx": durations[0], "cz": durations[1],
                          "measure": durations[2]}

        # ── Pinned placement (parity): when --physical-qubits is given, resolve
        #    it through the SAME solver seam the sweep uses
        #    (resolve_placements -> validated Placement), then derive
        #    phys_qubits/phys_edges exactly as the sweep parent does
        #    (sweep_engine _execute_byo_group). This drives prepare_simulation's
        #    device-cal branch onto a fixed initial_layout = the canonical
        #    placement, so transpile is deterministic (no free Sabre) and both
        #    arms sit on identical physical qubits. Absent the flag, prep keeps
        #    its default (free-layout) behaviour -- unchanged. ──
        prep_phys_qubits = None
        prep_phys_edges = None
        if physical_qubits is not None and noise_source == "device-calibrated":
            if GeneralPlacementSolver is None or extract_connectivity is None:
                raise RuntimeError("placement seam not importable")
            connectivity = extract_connectivity(circuits[-1])
            reg = PluginRegistry()
            reg.discover()
            cal_json = json.load(open(calibration_path))
            adapter = reg.get_calibration_adapter(cal_json.get("adapter", "iqm_v2"))
            device_cal = adapter.load(calibration_path)
            solver = GeneralPlacementSolver()
            solver.add_device(device_cal)
            placements = solver.resolve_placements(
                circuit_edges=connectivity,
                circuit_qubits=num_qubits,
                device_id=device_cal.device_id,
                strategy="max_fidelity",
                manual_qubit_name_lists=[physical_qubits],
            )
            placement = placements[0]
            prep_phys_qubits = [placement.qubit_mapping[i] for i in range(num_qubits)]
            prep_phys_edges = [
                (prep_phys_qubits[a], prep_phys_qubits[b]) for (a, b) in connectivity
            ]
            print(f"{tag} pinned placement: {prep_phys_qubits} (solver bypassed)")

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
            physical_qubits=prep_phys_qubits,
            physical_edges=prep_phys_edges,
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
                # Logical-vs-native circuit complexity (device-calibrated only;
                # null for noiseless / iqm-fake-backend, which do not native-
                # compile via the device path). Computed in the prepare() seam.
                "circuit_metrics": prep.info.get("circuit_metrics") or None,
                # IQM fake backend class name when applicable, else null.
                "fake_backend": prep.info.get("fake_backend"),
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


def _parse_physical_qubits_cli(raw):
    """Normalize the --physical-qubits CLI value into a name list (or None).

    Comma-separated, each name whitespace-stripped, blanks dropped. None or an
    empty string -> None (the free-layout default). A non-empty but all-blank
    value (e.g. " , ") yields [] -- deliberately NOT None -- so a malformed flag
    fails loud on the placement length check rather than silently free-layouting.
    Single source of truth for the runner's --physical-qubits parse; the parity
    guard test exercises this exact function.
    """
    if not raw:
        return None
    return [q.strip() for q in raw.split(",") if q.strip()]


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
    parser.add_argument(
        "--disorder-file", default=None,
        help="Path to a banked disorder JSON (source:file). When given, the "
             "runner consumes per-instance hz_angles/Jzz_angles/init_bit_array "
             "from the file via the sweep's resolve_disorder (RNG-free, with "
             "the same fail-loud assertions), instead of drawing them. Default "
             "(absent): draw from the master-seed-derived stream (prior "
             "behaviour). Use this to share one disorder realization with a "
             "BYO sweep arm for bit-level parity.")
    parser.add_argument(
        "--physical-qubits", default=None,
        help="Comma-separated physical qubit names (e.g. QB11,QB5,...), logical "
             "i -> the i-th name, pinning the device-calibrated placement via "
             "the solver seam (solver bypassed, list validated as a calibrated "
             "chain). Default (absent): solver/free-layout placement (prior "
             "behaviour). device-calibrated only.")
    args = parser.parse_args()

    physical_qubits = _parse_physical_qubits_cli(args.physical_qubits)
    if args.disorder_file and not os.path.isfile(args.disorder_file):
        sys.exit(f"ERROR: --disorder-file not found: {args.disorder_file}")

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
    print(f"disorder      : {args.disorder_file if args.disorder_file else 'drawn (master-seed)'}")
    print(f"physical_qubits: {physical_qubits if physical_qubits else 'solver/free-layout'}")
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
         args.t2_mode, durations, args.iqm_device, noise_spec, master_seed,
         args.disorder_file, physical_qubits)
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
