# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Offline unit tests for the generic cross-node shard layer (sweep/fanout.py).

Pure-Python (no aer/rustworkx/h5py): the shard is integer index math + SLURM env
parsing. Covers the indexing Red flagged (the non-dividing 8/3 case where an
off-by-one would hide), the partition validity (every unit assigned exactly
once), and the §3.3 env-co-residency invariant — including a test that
demonstrates plain round-robin SEGREGATES envs at nranks=2 and that stratified
round-robin fixes it.
"""

import importlib.util
import os

# Load sweep/fanout.py directly by path. The module is pure stdlib (index math +
# SLURM env parsing); importing it via the package would pull sweep/__init__ ->
# mixed_packing -> qiskit, an unnecessary heavy dep for testing pure logic. This
# keeps the test runnable offline (no aer/qiskit) and on LUMI alike.
_FANOUT = os.path.join(
    os.path.dirname(__file__), "..", "..", "src",
    "lumi_hpc_qc", "sweep", "fanout.py",
)
_spec = importlib.util.spec_from_file_location("_fanout_under_test", _FANOUT)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
resolve_rank_nranks = _mod.resolve_rank_nranks
shard = _mod.shard
shard_indices = _mod.shard_indices


def _partition(num_units, nranks, strata=None):
    return [shard_indices(num_units, r, nranks, strata) for r in range(nranks)]


def test_round_robin_basic():
    assert shard_indices(8, 0, 3) == [0, 3, 6]
    assert shard_indices(8, 1, 3) == [1, 4, 7]
    assert shard_indices(8, 2, 3) == [2, 5]


def test_non_dividing_slice_sizes():
    # Red's specific ask: 8 units / 3 ranks -> 3/3/2, the case where an
    # off-by-one in `i mod nranks` would hide.
    parts = _partition(8, 3)
    assert [len(p) for p in parts] == [3, 3, 2]


def test_partition_is_exact():
    # Across all ranks every index appears exactly once (no loss, no dup).
    for num_units in (0, 1, 7, 8, 240):
        for nranks in (1, 2, 3, 5, 8):
            parts = _partition(num_units, nranks)
            flat = sorted(i for p in parts for i in p)
            assert flat == list(range(num_units))
            # disjoint
            seen = set()
            for p in parts:
                assert not (set(p) & seen)
                seen |= set(p)


def test_balanced_to_within_one():
    parts = _partition(240, 7)
    sizes = sorted(len(p) for p in parts)
    assert sizes[-1] - sizes[0] <= 1


def test_single_rank_is_noop():
    items = list("abcdef")
    assert shard(items, 0, 1) == items


def test_plain_round_robin_segregates_envs_at_nranks_2():
    # Flat order: for placement -> for seed -> for env, so envs alternate.
    # 2 placements x 2 seeds x 2 envs = 8 units, env interleaved.
    envs = ["noiseless", "device_calibrated"] * 4  # n,d,n,d,n,d,n,d
    r0 = shard_indices(8, 0, 2)            # plain (no strata)
    r1 = shard_indices(8, 1, 2)
    r0_envs = {envs[i] for i in r0}
    r1_envs = {envs[i] for i in r1}
    # The PROBLEM: plain round-robin at nranks=2 puts one env entirely on each
    # rank (all noiseless on rank0, all noisy on rank1).
    assert r0_envs == {"noiseless"}
    assert r1_envs == {"device_calibrated"}


def test_stratified_round_robin_keeps_both_envs_per_rank():
    # The FIX: passing env as strata keeps every rank env-mixed at any nranks,
    # including nranks=2 where plain round-robin segregated (§3.3 invariant).
    envs = ["noiseless", "device_calibrated"] * 4
    for nranks in (2, 3, 4):
        for r in range(nranks):
            idxs = shard_indices(8, r, nranks, strata=envs)
            rank_envs = {envs[i] for i in idxs}
            # Each rank that receives any units receives BOTH envs (balanced
            # stratification: 4 of each env across nranks ranks).
            if idxs:
                assert rank_envs == {"noiseless", "device_calibrated"}, (
                    f"nranks={nranks} rank={r} got {rank_envs}"
                )
    # And the stratified partition is still exact.
    parts = _partition(8, 2, strata=envs)
    assert sorted(i for p in parts for i in p) == list(range(8))


def test_stratified_balances_each_stratum():
    # 12 placements x 10 seeds x 2 envs = 240, env-interleaved; 2 ranks.
    envs = ["noiseless", "device_calibrated"] * 120
    r0 = shard_indices(240, 0, 2, strata=envs)
    r1 = shard_indices(240, 1, 2, strata=envs)
    for env in ("noiseless", "device_calibrated"):
        c0 = sum(1 for i in r0 if envs[i] == env)
        c1 = sum(1 for i in r1 if envs[i] == env)
        assert abs(c0 - c1) <= 1 and c0 + c1 == 120


def test_resolve_rank_nranks_node_based():
    env = {"SLURM_NNODES": "4", "SLURM_NODEID": "2"}
    assert resolve_rank_nranks(env) == (2, 4)


def test_resolve_rank_nranks_task_fallback():
    env = {"SLURM_NTASKS": "3", "SLURM_PROCID": "1"}
    assert resolve_rank_nranks(env) == (1, 3)


def test_resolve_rank_nranks_default_single():
    assert resolve_rank_nranks({}) == (0, 1)


def test_validation_errors():
    for bad in (
        lambda: shard_indices(8, 0, 0),       # nranks < 1
        lambda: shard_indices(8, 3, 3),       # rank == nranks
        lambda: shard_indices(8, -1, 3),      # rank < 0
        lambda: shard_indices(8, 0, 2, ["x"]),  # strata len mismatch
    ):
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"  PASS {fn.__name__}")
    print(f"FANOUT SHARD: ALL {len(fns)} CHECKS PASSED")
