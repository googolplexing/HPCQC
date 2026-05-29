# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""W1.3 — BYO sweep engine-internal worker (per-(seed, placement, env)).

The forkserver child process called by ``_execute_byo_group`` for one work
unit. Mirrors the runner's ``floquet_runner.run_one_instance`` (the model
endorsed by RED-RESP-W1-PARALLELISM-AND-OOM-ROOTCAUSE-v1.4 Asks 1+2 +
Q1 ACCEPT): lean, single-arm, single-thread Aer, parent-serial writes.

Module-level imports
--------------------

Heavy imports (qiskit-aer via prepare_simulation, qiskit via the factory
loader, numpy via byo_observable) live at module level. There is NO
``set_forkserver_preload`` anywhere in the engine or runner, so the
forkserver *server* is a lean exec'd interpreter that does not import this
module — each forked worker imports the heavy stack *after* the fork. The
memory saving is therefore automatic OS-level shared-library (``.so``) page
sharing across workers that map the same libraries, NOT a preloaded
Python-heap copy-on-write of the resident stack. This is empirically
harmless and runner-identical (proven at 40-way: runner cgroup ``MaxRSS``
20.18 GiB, RED-RESP-W1 v1.4 §1.3, while the prior shell fork-per-seed
workaround OOM'd at 98 s — LUMI job 18899724). It is, however, load-bearing
for the W1.4 cap: because each worker carries its OWN full resident
footprint post-fork (no near-zero marginal heap), the allocation-aware cap
sizes every worker at its full ``VmHWM``. (Corrected per
RED-RESP-W1.3-VERIFY-AND-W1.4-CAP-RULINGS NF5 / D6-i; the previous wording
asserted a preloaded-heap CoW economy the code does not implement.)

Picklability
------------

``WorkerArgs`` and ``WorkerResult`` are plain dataclasses of picklable
primitives (int, str, list, dict, None). No QuantumCircuit, no NoiseConfig,
no Placement objects cross the process boundary. The worker rebuilds its
seed's grid circuits from the factory at invocation time — paying
microseconds of Python construction per unit in exchange for not pickling
QuantumCircuit objects across the fork.

F6 invariant (RED-RESP-W1 v1.4)
-------------------------------

``seed_simulator = resolve_instance_seed(master_seed, seed)`` is recomputed
per-worker; ``env_name`` / ``env_source`` (the arm) are NOT part of the
derivation. The parent identity-shares ``disorder_instance`` across both
arms of a seed (preserved at expansion time, ``_expand_byo_experiment``);
workers MUST NOT redraw disorder per arm — they receive it and rebuild the
circuit deterministically. Per-arm units of the same seed therefore yield
the identical disorder signature; the W1 canary verifies byte-match
against the in-tree oracle (``evidence/W1/gate2_canary/sha256_oracle.txt``)
at the 2-seed scale, and the W6 canary will verify it at full 40-seed scale.

Error carrying
--------------

A worker that fails returns a ``WorkerResult(..., error=str(e))`` rather
than raising — propagating an exception across ``Pool.map`` poisons the
pool and aborts every sibling unit mid-flight (the runner pattern).
``_execute_byo_group`` inspects ``error`` after ``pool.map`` returns and
fails the group loudly if any unit failed.
"""

from __future__ import annotations

import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

# Heavy imports. Workers import these FRESH after the forkserver fork (there
# is no set_forkserver_preload); the saving is automatic OS-level .so page
# sharing, NOT preloaded-heap CoW. See the module docstring + NF5 / D6-i.
import numpy as np  # noqa: F401  (transitive: prepare_simulation, byo_observable)
from lumi_hpc_qc.backends.prepare import prepare_simulation
from lumi_hpc_qc.sweep.byo_observable import (
    get_autocorrelation,
    resolve_instance_seed,
)
from lumi_hpc_qc.sweep.byo_sweep import assemble_build_kwargs
from lumi_hpc_qc.sweep.circuit_loader import load_circuit
from lumi_hpc_qc.sweep.worker_cap import read_vmhwm_kib  # D1: stdlib-only peak probe


@dataclass
class WorkerArgs:
    """Picklable arguments for a single (seed, placement, env) worker unit.

    All fields are picklable primitives — no QuantumCircuit, NoiseConfig, or
    Placement objects. The worker reloads the circuit factory and rebuilds
    the seed's grid circuits from these primitives at invocation time.

    The parent constructs one WorkerArgs per (seed, placement, env) triple
    in ``_execute_byo_group``. ``placement_phys_qubits`` /
    ``placement_phys_edges`` come from the parent's already-solved Placement
    (W1.1's deterministic ordering applies); they cross the boundary as
    plain lists/tuples.
    """
    # Identity
    seed: int
    env_name: str                                  # e.g. "device_calibrated", "noiseless"
    env_source: str                                # e.g. "device_calibrated", "channels"
    master_seed: int | None                        # for resolve_instance_seed; None -> entropy
    placement_id: int                              # for provenance

    # Placement (already solved by parent; deterministic per W1.1)
    placement_phys_qubits: list[str]               # logical -> physical, ordered
    placement_phys_edges: list[tuple[str, str]]    # for device_calibrated F5a noise composition

    # Execution
    calibration_path: str
    shots: int
    optimization_level: int                        # CFG-2 (W1.2); already resolved (default 3)
    qsize: int

    # Factory invocation (rebuilt in-worker — no QuantumCircuit pickling)
    factory_script: str
    factory_function: str
    fixed_params: dict[str, Any] = field(default_factory=dict)
    disorder_instance: dict[str, Any] = field(default_factory=dict)
    disorder_gates: tuple[str, ...] = ("rz", "rzz")

    # Observable + grid
    init_bit_array: list[int] = field(default_factory=list)
    primary_axis: str = "num_kicks"
    grid_points_sorted: list[dict[str, Any]] = field(default_factory=list)

    # Guardrail flag — parent decides (F5a / D3a guardrail: true only on
    # device_calibrated records that ran without per-placement composition).
    noise_placement_independent: bool = False


@dataclass
class WorkerResult:
    """Picklable result returned by ``run_one_unit``.

    The parent assembles these into ``byo_results`` dicts that are
    byte-format-identical to what the pre-W1 serial path produced —
    preserving the downstream HDF5 write + .dat aggregation contract
    (D3.4c / RED-RESP-D3.4C).
    """
    # Identity (echoed from WorkerArgs for the parent's bookkeeping)
    seed: int
    env_name: str
    env_source: str
    placement_id: int
    physical_qubit_set: list[str]

    # Result
    num_kicks: list[int]                           # the seed's grid axis values, ascending
    autocorrelator: list[float]                    # one float per grid point
    shots: int
    seed_simulator: int                            # = resolve_instance_seed(master_seed, seed)
    master_seed: int | None
    optimization_level: int

    # Guardrail / provenance pass-through
    noise_placement_independent: bool

    # Observability
    runtime_s: float = 0.0
    # D1 (W1.4): kernel high-water mark (VmHWM) in KiB. The worker reads its
    # OWN peak at the end of run_one_unit and returns it so the parent can size
    # the main pool from a device_calibrated probe without racing /proc/<child>.
    # 0 when unread (error path / probe not taken).
    peak_rss_kib: int = 0

    # Error carrying — None on success; populated string on failure.
    # See module docstring on why we don't raise.
    error: str | None = None


def run_one_unit(args: WorkerArgs) -> WorkerResult:
    """Per-(seed, placement, env) worker. Mirrors ``floquet_runner.run_one_instance``.

    Single-arm per worker: rebuild the seed's grid circuits via the factory,
    transpile at ``args.optimization_level`` (CFG-2), prepare the simulation
    with the right noise source (and per-placement physical_qubits/edges for
    ``device_calibrated`` via the F5a seam), run one Aer call over the
    kick-list, compute the autocorrelator vector, return it.

    All exceptions are captured into ``WorkerResult.error`` rather than
    raised, to avoid poisoning the pool (see module docstring).
    """
    # ── W1.3: pin the per-worker execution environment BEFORE any qiskit /
    #    Aer call. This is the definitive guard, executed inside the worker
    #    process; ``parallel_map`` / OpenMP read these env vars lazily (at the
    #    first parallel region, i.e. first compute), so setting them here —
    #    after the module-level heavy imports but before load_circuit /
    #    prepare_simulation / simulator.run — takes effect. Mirrors
    #    floquet_runner.run_one_instance:289.
    #
    #    QISKIT_IN_PARALLEL=TRUE — REQUIRED for the device_calibrated arm. That
    #    arm's NoiseModel carries a custom pass (RelaxationNoisePass appended to
    #    ``nm._custom_noise_passes`` in backends/prepare.py); Aer applies custom
    #    noise passes via qiskit's transpiler at ``simulator.run()`` time, over
    #    the whole grid-circuit list. With this UNSET, qiskit's ``parallel_map``
    #    tries to spawn its own worker Pool to parallelize across the circuit
    #    list — but this function already runs inside a daemonic forkserver Pool
    #    child, and Python forbids daemons from having children, raising
    #    ``AssertionError: daemonic processes are not allowed to have children``.
    #    Setting TRUE makes ``parallel_map`` run serially in-process. The
    #    noiseless arm has no custom pass, so it never hit this — which is why
    #    the bug presented as device_calibrated-only. (Was previously masked by
    #    the os-shadow UnboundLocalError, which aborted before the Pool existed.)
    #
    #    OMP_NUM_THREADS=1 — single-thread the worker's OpenMP/BLAS so N
    #    co-resident workers on one node do not oversubscribe cores (proposal
    #    v1.1: "per worker ... plus OMP_NUM_THREADS=1"). Aer is already pinned
    #    single-thread via _AER_SINGLE_THREAD; this covers numpy/BLAS and any
    #    other OpenMP consumer in the worker.
    os.environ["QISKIT_IN_PARALLEL"] = "TRUE"
    os.environ["OMP_NUM_THREADS"] = "1"

    t0 = time.perf_counter()

    def _err_result(msg: str) -> WorkerResult:
        return WorkerResult(
            seed=args.seed,
            env_name=args.env_name,
            env_source=args.env_source,
            placement_id=args.placement_id,
            physical_qubit_set=args.placement_phys_qubits,
            num_kicks=[],
            autocorrelator=[],
            shots=args.shots,
            seed_simulator=0,
            master_seed=args.master_seed,
            optimization_level=args.optimization_level,
            noise_placement_independent=args.noise_placement_independent,
            runtime_s=time.perf_counter() - t0,
            error=msg,
        )

    try:
        # ── Rebuild the seed's grid circuits via the factory. Mirrors the
        #    parent's _build_byo_circuit (Gap A seam): same assemble_build_kwargs
        #    fixed ∪ disorder ∪ grid_point, same load_circuit signature. ──
        built = []
        for grid_point in args.grid_points_sorted:
            build_kwargs = assemble_build_kwargs(
                args.fixed_params, args.disorder_instance, grid_point,
            )
            loaded = load_circuit(
                script_file=args.factory_script,
                script_function=args.factory_function,
                script_params=build_kwargs,
            )
            built.append(loaded.circuit)

        # ── F6 invariant: seed_simulator = resolve_instance_seed(master_seed,
        #    seed). Arm (env_name / env_source) is NOT part of the derivation.
        #    Both arms of seed N must therefore share this value, and (because
        #    the parent identity-shares disorder_instance across arms) sample
        #    the same disorder realization under different noise. The W1
        #    canary asserts byte-match against the in-tree oracle. ──
        inst_seed = resolve_instance_seed(args.master_seed, args.seed)

        # ── Source-name mapping mirrors the pre-W1 parent code: NoiseConfig
        #    "device_calibrated" (underscore) -> prepare_simulation's
        #    VALID_SOURCES "device-calibrated" (hyphen); other names pass
        #    through unchanged. ──
        src = ("device-calibrated"
               if args.env_source == "device_calibrated"
               else args.env_name)

        prep_kwargs = dict(
            calibration_path=args.calibration_path,
            num_qubits=args.qsize,
            optimization_level=args.optimization_level,
            num_processes=1,                 # CFG-2 Q3: tied to single-thread worker
            verbose=False,
        )
        if args.env_source == "device_calibrated":
            prep_kwargs.update(
                physical_qubits=args.placement_phys_qubits,
                physical_edges=args.placement_phys_edges,
            )

        prep = prepare_simulation(built, src, **prep_kwargs)

        # ── Single Aer run over the seed's kick-list (mirrors the runner's
        #    one-job-per-instance pattern, which is what makes the gate-2
        #    reproduction bit-exact against the banked reference). ──
        job = prep.simulator.run(
            prep.run_circuits, shots=args.shots, memory=True,
            seed_simulator=inst_seed,
        )
        result = job.result()

        autocorr = [
            get_autocorrelation(
                result.get_counts(i), args.init_bit_array, args.qsize,
            )
            for i in range(len(built))
        ]
        num_kicks = [g[args.primary_axis] for g in args.grid_points_sorted]

        # D1: read this worker's OWN kernel high-water mark at the very end —
        # after build/transpile/run, so it captures the unit's true peak — and
        # return it. The parent sizes the main pool from a device_calibrated
        # probe's peak (no /proc/<child> race). 0 if /proc + getrusage both fail.
        peak_rss_kib = read_vmhwm_kib("self") or 0

        return WorkerResult(
            seed=args.seed,
            env_name=args.env_name,
            env_source=args.env_source,
            placement_id=args.placement_id,
            physical_qubit_set=args.placement_phys_qubits,
            num_kicks=num_kicks,
            autocorrelator=autocorr,
            shots=args.shots,
            seed_simulator=int(inst_seed) if inst_seed is not None else 0,
            master_seed=args.master_seed,
            optimization_level=args.optimization_level,
            noise_placement_independent=args.noise_placement_independent,
            runtime_s=time.perf_counter() - t0,
            peak_rss_kib=peak_rss_kib,
            error=None,
        )

    except Exception as e:
        return _err_result(
            f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        )
