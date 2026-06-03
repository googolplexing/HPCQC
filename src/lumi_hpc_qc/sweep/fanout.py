# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Generic, sweep-type-agnostic work-unit sharding for across-node fan-out.

This is the GENERIC layer of the Workstream-B cross-node fan-out
(RED-RULING-WORKSTREAM-B-CROSS-NODE-FANOUT §4 directive): it round-robin-shards a
flat list of work units across SLURM node-ranks. It knows NOTHING about what a
unit is — the same shard serves the BYO floquet path, the hamiltonian battery
path, and any future flat-work-unit sweep. The per-path reduction at merge is a
separate pluggable layer; this module is purely the index partition + the
SLURM-allocation lookup.

Determinism / byte-identity: the shard is a pure function of
``(num_units, rank, nranks[, strata])``. A unit's output is rank-invariant
(``resolve_instance_seed`` depends only on ``master_seed`` + ``seed``, never on
rank/wave/dispatch order), so which rank runs a unit cannot change its result —
the byte-identity gate rests on that, and this partition only decides WHERE each
unit runs, not what it computes.

Stratification (§3.3 env-co-residency invariant): plain ``i mod nranks`` can
segregate a stratum onto one rank when ``nranks`` aligns with that stratum's
period in the flat order. Concretely, the flat list is built
``for placement → for seed → for env`` so the two noise envs alternate every
index; with ``nranks == 2`` plain round-robin puts every noiseless unit on rank 0
and every noisy unit on rank 1 — envs segregated across nodes, the heavy noisy
units all on one node, and the "both envs co-resident per rank" invariant broken.
Passing ``strata`` (one key per unit, e.g. the env name) makes the shard
round-robin WITHIN each stratum, so every rank gets a balanced share of each
stratum and stays mixed regardless of ``nranks``. The layer never interprets a
stratum key; the caller supplies env (or any key), so genericity is preserved.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from collections.abc import Hashable, Mapping, Sequence
from typing import TypeVar

T = TypeVar("T")

_DEFAULT_RANK = 0
_DEFAULT_NRANKS = 1


def _validate(rank: int, nranks: int) -> None:
    if nranks < 1:
        raise ValueError(f"nranks must be >= 1, got {nranks}")
    if not (0 <= rank < nranks):
        raise ValueError(f"rank must be in [0, {nranks}), got {rank}")


def resolve_rank_nranks(env: Mapping[str, str] | None = None) -> tuple[int, int]:
    """``(rank, nranks)`` for this process from the SLURM allocation.

    Prefers ``SLURM_NODEID`` / ``SLURM_NNODES`` (one rank per node — the proposed
    ``--nodes=N --ntasks-per-node=1`` job shape). Falls back to ``SLURM_PROCID`` /
    ``SLURM_NTASKS`` (or ``SLURM_NPROCS``), then to single-rank ``(0, 1)`` for a
    local or single-node run. Single-rank is the no-op shard: ``shard()`` returns
    the whole list unchanged, so the certified single-node path is byte-untouched.
    """
    env = os.environ if env is None else env
    nn = env.get("SLURM_NNODES")
    nid = env.get("SLURM_NODEID")
    if nn is not None and nid is not None:
        nranks, rank = int(nn), int(nid)
    else:
        nt = env.get("SLURM_NTASKS") or env.get("SLURM_NPROCS")
        pid = env.get("SLURM_PROCID")
        if nt is not None and pid is not None:
            nranks, rank = int(nt), int(pid)
        else:
            nranks, rank = _DEFAULT_NRANKS, _DEFAULT_RANK
    _validate(rank, nranks)
    return rank, nranks


def shard_indices(
    num_units: int,
    rank: int,
    nranks: int,
    strata: Sequence[Hashable] | None = None,
) -> list[int]:
    """This rank's indices into a flat list of ``num_units`` (round-robin).

    Without ``strata``: ``unit i -> rank (i mod nranks)``. Round-robin (not
    contiguous blocks) keeps slice sizes balanced (differ by <= 1).

    With ``strata`` (one opaque key per unit, ``len == num_units``): round-robin
    WITHIN each stratum, so every rank gets a balanced share of each stratum and
    stays stratum-mixed regardless of ``nranks`` (the §3.3 env-co-residency
    invariant — pass env names as strata). The returned indices are sorted
    ascending for deterministic downstream ordering.

    In both cases the partition is exact: across all ranks every index is
    assigned to exactly one rank.
    """
    _validate(rank, nranks)
    if num_units < 0:
        raise ValueError(f"num_units must be >= 0, got {num_units}")
    if strata is None:
        return list(range(rank, num_units, nranks))
    if len(strata) != num_units:
        raise ValueError(
            f"strata length {len(strata)} != num_units {num_units}"
        )
    # Group indices by stratum (insertion-ordered for determinism), then take
    # every nranks-th within each stratum starting at `rank`.
    by_stratum: OrderedDict[Hashable, list[int]] = OrderedDict()
    for i, key in enumerate(strata):
        by_stratum.setdefault(key, []).append(i)
    selected: list[int] = []
    for idxs in by_stratum.values():
        selected.extend(idxs[rank::nranks])
    selected.sort()
    return selected


def shard(
    items: Sequence[T],
    rank: int,
    nranks: int,
    strata: Sequence[Hashable] | None = None,
) -> list[T]:
    """This rank's slice of a flat work-unit list. Generic over the unit type —
    the layer never inspects a unit (``strata``, if given, is supplied by the
    caller and treated as opaque keys)."""
    return [items[i] for i in shard_indices(len(items), rank, nranks, strata)]
