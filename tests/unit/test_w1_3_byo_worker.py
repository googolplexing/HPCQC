# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""W1.3 — byo_worker module invariants.

Tests:
  (1) WorkerArgs and WorkerResult are picklable (forkserver Pool requires it;
      a non-picklable arg silently hangs the pool at runtime — fast-fail in
      unit tests instead).
  (2) F6 invariant — arm-independent seed derivation.
      resolve_instance_seed(master_seed, seed) does NOT take an arm argument,
      so two WorkerArgs that differ only in env_name/env_source must produce
      the same seed_simulator.
  (3) F6 invariant — disorder identity-sharing across arms.
      WorkerArgs constructed for the two arms of a seed (same master_seed,
      same seed, same disorder_instance dict) carry the SAME disorder_instance
      and disorder_gates; the worker therefore rebuilds the SAME disorder-
      bearing gate angles (extract_disorder_signature would return equal
      signatures — verified end-to-end by the W1 canary, not here, since the
      container needed for circuit construction is not available locally).
  (4) WorkerResult schema completeness — every field the parent's byo_results
      assembly reads (sweep_engine._execute_byo_group) is present on a freshly
      constructed WorkerResult.
  (5) Error-carrying path — when run_one_unit's protected region raises, the
      result carries the error string and identity fields rather than
      propagating the exception (which would poison the forkserver Pool).

Tests (1), (2), (4), (5) are pure-Python and run anywhere. Test (3) verifies
the data-level F6 contract structurally; the end-to-end physical-byte-match
verification belongs to the LUMI canary (tests/slurm_w1_canary.sh) which
asserts sha256_oracle.txt agreement at the 2-seed scale.

Per RED-RESP-W1-PARALLELISM-AND-OOM-ROOTCAUSE-v1.4 F6: "Workers must be handed
the resolved disorder_instance + master_seed and never re-draw disorder per
arm. Both arms of seed N sample the same realization under different noise."
"""

from __future__ import annotations

import pickle

from lumi_hpc_qc.sweep.byo_worker import (
    WorkerArgs,
    WorkerResult,
    run_one_unit,
)
from lumi_hpc_qc.sweep.byo_observable import resolve_instance_seed


# ── Sample fixtures (no Qiskit objects — pure picklable primitives) ────────

def _sample_args(env_name: str = "device_calibrated",
                 env_source: str = "device_calibrated",
                 seed: int = 0,
                 master_seed: int = 12345,
                 disorder_instance: dict | None = None) -> WorkerArgs:
    """A representative WorkerArgs with all required fields populated."""
    if disorder_instance is None:
        disorder_instance = {
            "init_bit_array": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            "hz_angles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "Jzz_angles": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
        }
    return WorkerArgs(
        seed=seed,
        env_name=env_name,
        env_source=env_source,
        master_seed=master_seed,
        placement_id=0,
        placement_phys_qubits=[f"QB{i}" for i in range(10)],
        placement_phys_edges=[(f"QB{i}", f"QB{i+1}") for i in range(9)],
        calibration_path="/tmp/fake_cal.json",
        shots=1000,
        optimization_level=3,
        qsize=10,
        factory_script="examples/byo/floquet_dtc.py",
        factory_function="build_circuit",
        fixed_params={"epsilon": 0.05},
        disorder_instance=disorder_instance,
        disorder_gates=("rz", "rzz"),
        init_bit_array=[0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        primary_axis="num_kicks",
        grid_points_sorted=[{"num_kicks": k} for k in (1, 10, 60)],
        noise_placement_independent=True,
    )


def _sample_result() -> WorkerResult:
    """A representative successful WorkerResult."""
    return WorkerResult(
        seed=0,
        env_name="device_calibrated",
        env_source="device_calibrated",
        placement_id=0,
        physical_qubit_set=[f"QB{i}" for i in range(10)],
        num_kicks=[1, 10, 60],
        autocorrelator=[0.99, 0.42, 0.11],
        shots=1000,
        seed_simulator=987654321,
        master_seed=12345,
        optimization_level=3,
        noise_placement_independent=True,
        runtime_s=12.3,
        error=None,
    )


# ── (1) Picklability ──────────────────────────────────────────────────────

def test_worker_args_picklable():
    """forkserver Pool.map requires picklable args; non-picklable hangs the
    pool silently at runtime — surface here as a fast-fail unit test."""
    args = _sample_args()
    blob = pickle.dumps(args)
    restored = pickle.loads(blob)
    assert restored.seed == args.seed
    assert restored.env_name == args.env_name
    assert restored.master_seed == args.master_seed
    assert restored.placement_phys_qubits == args.placement_phys_qubits
    assert restored.disorder_instance == args.disorder_instance
    assert restored.grid_points_sorted == args.grid_points_sorted


def test_worker_result_picklable():
    """Same contract for the return value from each worker."""
    result = _sample_result()
    blob = pickle.dumps(result)
    restored = pickle.loads(blob)
    assert restored.seed == result.seed
    assert restored.autocorrelator == result.autocorrelator
    assert restored.seed_simulator == result.seed_simulator
    assert restored.master_seed == result.master_seed
    assert restored.error is None


def test_worker_result_with_error_picklable():
    """The error-carrying path must also pickle (parent receives it via map)."""
    result = WorkerResult(
        seed=3, env_name="device_calibrated", env_source="device_calibrated",
        placement_id=0, physical_qubit_set=["QB0"], num_kicks=[],
        autocorrelator=[], shots=1000, seed_simulator=0, master_seed=42,
        optimization_level=3, noise_placement_independent=True,
        runtime_s=0.5, error="RuntimeError: simulated failure\n  traceback...",
    )
    restored = pickle.loads(pickle.dumps(result))
    assert restored.error is not None
    assert "simulated failure" in restored.error


# ── (2) F6 invariant — arm-independent seed derivation ────────────────────

def test_seed_simulator_is_arm_independent():
    """resolve_instance_seed takes (master_seed, seed); arm is NOT part of
    the derivation. Two WorkerArgs that differ only in env must compute the
    same seed_simulator."""
    args_a = _sample_args(env_name="noiseless", env_source="channels")
    args_b = _sample_args(env_name="device_calibrated", env_source="device_calibrated")
    seed_a = resolve_instance_seed(args_a.master_seed, args_a.seed)
    seed_b = resolve_instance_seed(args_b.master_seed, args_b.seed)
    assert seed_a == seed_b, (
        f"F6 violated: arm changes seed_simulator. "
        f"noiseless -> {seed_a}, device_calibrated -> {seed_b}"
    )


def test_seed_simulator_varies_with_seed():
    """Sanity guard for the test above: different seeds DO give different
    seed_simulator values (otherwise the F6 test would trivially pass)."""
    s0 = resolve_instance_seed(12345, 0)
    s1 = resolve_instance_seed(12345, 1)
    assert s0 != s1


def test_seed_simulator_varies_with_master_seed():
    """Sanity guard: different master_seeds DO give different seed_simulator
    values (otherwise instances would alias across runs)."""
    a = resolve_instance_seed(12345, 0)
    b = resolve_instance_seed(99999, 0)
    assert a != b


def test_seed_simulator_none_master_seed_returns_none():
    """master_seed=None -> entropy (not reproducible); resolve returns None.
    Worker propagates this and seed_simulator field becomes 0 on the result
    (per the docstring's int-or-zero contract)."""
    assert resolve_instance_seed(None, 0) is None


# ── (3) F6 invariant — disorder identity-sharing across arms ──────────────

def test_disorder_instance_identity_shared_across_arms():
    """Constructed for the same (master_seed, seed) with the same
    disorder_instance dict, the two arms' WorkerArgs carry IDENTICAL
    disorder content. The parent identity-shares disorder_instance at
    expansion (§7.5.4); the worker must NOT re-draw."""
    shared = {
        "init_bit_array": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
        "hz_angles": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "Jzz_angles": [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09],
    }
    args_noiseless = _sample_args(
        env_name="noiseless", env_source="channels",
        disorder_instance=shared,
    )
    args_devcal = _sample_args(
        env_name="device_calibrated", env_source="device_calibrated",
        disorder_instance=shared,
    )
    # Same disorder data on both arms
    assert args_noiseless.disorder_instance == args_devcal.disorder_instance
    # Same disorder-bearing gate set on both arms
    assert args_noiseless.disorder_gates == args_devcal.disorder_gates
    # Same fixed parameters on both arms
    assert args_noiseless.fixed_params == args_devcal.fixed_params
    # Same grid points on both arms (factory builds same circuit structure)
    assert args_noiseless.grid_points_sorted == args_devcal.grid_points_sorted


# ── (4) WorkerResult schema completeness ──────────────────────────────────

def test_worker_result_carries_every_field_parent_reads():
    """The parent (sweep_engine._execute_byo_group post-W1.3) reads these
    fields off WorkerResult when assembling the byo_results dict. Pinning the
    contract here so a future field rename surfaces immediately."""
    r = _sample_result()
    # Every dict key in the parent's assembly:
    expected_attrs = (
        "seed", "env_name", "env_source", "placement_id",
        "physical_qubit_set", "num_kicks", "autocorrelator", "shots",
        "seed_simulator", "master_seed", "optimization_level",
        "noise_placement_independent",
    )
    for attr in expected_attrs:
        assert hasattr(r, attr), f"WorkerResult missing required field {attr!r}"


def test_worker_result_includes_runtime_and_error():
    """Observability fields used by W1.5 footer and W1.3 fail-loud."""
    r = _sample_result()
    assert hasattr(r, "runtime_s")
    assert isinstance(r.runtime_s, (int, float))
    assert hasattr(r, "error")
    # On the success path, error is None.
    assert r.error is None


# ── (5) Error-carrying path ───────────────────────────────────────────────

def test_run_one_unit_returns_error_result_on_failure():
    """A worker that raises inside the protected region must return a
    WorkerResult with `error` populated rather than letting the exception
    propagate. Use deliberately-broken args (script_file that does not
    exist) — load_circuit will raise; we want that captured."""
    args = _sample_args()
    # Replace the factory_script with a path that cannot exist
    args.factory_script = "/nonexistent/path/to/factory_circuit.py"
    args.factory_function = "build_circuit"
    r = run_one_unit(args)
    # Must NOT have raised; must return a WorkerResult with error populated.
    assert isinstance(r, WorkerResult)
    assert r.error is not None
    # Identity preserved so the parent's bounded error report can locate it
    assert r.seed == args.seed
    assert r.env_name == args.env_name
    assert r.placement_id == args.placement_id
    # Numeric defaults on the failure path
    assert r.autocorrelator == []
    assert r.num_kicks == []


def test_error_result_runtime_is_recorded():
    """Even on failure, runtime_s is populated (debugging info: how long did
    the worker take before failing?)."""
    args = _sample_args()
    args.factory_script = "/nonexistent/path/again.py"
    r = run_one_unit(args)
    assert r.runtime_s >= 0.0
