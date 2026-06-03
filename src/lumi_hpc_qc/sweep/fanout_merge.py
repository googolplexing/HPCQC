# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Generic, sweep-type-agnostic merge layer for across-node fan-out.

The merge half of the Workstream-B cross-node fan-out
(RED-RULING-WORKSTREAM-B-CROSS-NODE-FANOUT §4 directive). After per-rank jobs
each write their disjoint per-unit HDF5 groups, the merge:

  1. UNIONs the per-rank HDF5 files into one (disjoint per-unit groups -> a
     straight copy-in; shape-agnostic, works for any per-unit-group sweep);
  2. extracts the per-unit records via a per-path REDUCER (the only sweep-
     specific piece), groups them, ASSERTS each group's instance set is complete
     (RED-RULING §2 precondition — the short-count guard), then invokes the
     reducer's deferred aggregation per group;
  3. CONCATs the per-rank manifests.

Genericity boundary (the directive): the shard layer (fanout.py), the HDF5
union, the completeness-asserted dispatch driver, and the manifest concat are
GENERIC (built once, reused by any flat-work-unit sweep). The only path-specific
piece is the REDUCER — for BYO floquet it wraps ``aggregate_byo_autocorr``; the
hamiltonian battery path (and any future sweep) registers its own. The generic
``assert_complete_and_reduce`` driver never interprets a group key, instance key,
or payload; it only enforces completeness and dispatches.

Byte-identity: the reduce is the SAME certified function over the SAME complete
input as a single-node run (it is relocated to merge-time, not reimplemented),
and the per-unit HDF5 groups are unioned unchanged — so the merged result equals
a single-node run of the same units. The completeness assertion is what keeps
that true under a lost or partial shard: a partial series would silently average
to a wrong-but-plausible mean, so completeness is a PRECONDITION of the reduce.

This module imports only stdlib at module level; h5py / numpy / the manifest
class / the BYO aggregator are imported lazily inside the functions that need
them, so the pure dispatch + concat logic is testable without aer/h5py.
"""

from __future__ import annotations

import json
import os
from collections import OrderedDict
from collections.abc import Hashable, Iterable, Sequence
from typing import Any, Callable, Protocol


# ── the per-path reducer interface (the only sweep-specific seam) ───────────

class ShardReducer(Protocol):
    """Per-sweep-type plug for the generic merge. Implementations describe how to
    read per-unit records out of the merged HDF5, how to key them into groups +
    instances, what a complete group looks like, and how to aggregate one
    complete group's series. The generic driver supplies nothing sweep-specific."""

    def extract(self, merged_h5_path: str) -> Iterable[Any]:
        """Yield one opaque record per unit found in the merged HDF5."""

    def group_key(self, record: Any) -> Hashable:
        """The aggregation group a record belongs to (e.g. (placement, env))."""

    def instance_key(self, record: Any) -> Hashable:
        """The instance within a group (e.g. seed)."""

    def payload(self, record: Any) -> Any:
        """The data the reducer aggregates (e.g. the autocorrelator vector)."""

    def expected_instances(self, group_key: Hashable) -> Iterable[Hashable]:
        """The COMPLETE instance set this group must have (e.g. the seed_list).
        Completeness is asserted against this before aggregation."""

    def reduce(
        self, group_key: Hashable, series: list[tuple[Hashable, Any]], out_root: str
    ) -> None:
        """Aggregate one group's complete, instance-sorted series into outputs
        under out_root (e.g. aggregate_byo_autocorr -> aggregated_autocorr.dat)."""


# ── generic: completeness-asserted reduce dispatch (PURE; offline-tested) ───

def assert_complete_and_reduce(
    records: Iterable[Any], reducer: ShardReducer, out_root: str
) -> list[Hashable]:
    """Group ``records`` by ``reducer.group_key``, assert each group is COMPLETE
    against ``reducer.expected_instances`` (fail loud on any missing, extra, or
    duplicate instance — the short-count guard), then call ``reducer.reduce``
    with the instance-sorted series per group. Returns the reduced group keys.

    This is the load-bearing safety step (RED-RULING-WORKSTREAM-B §2): a partial
    series silently averaged would produce a wrong-but-plausible result, so
    completeness is a PRECONDITION of the reduce, not a post-hoc check. Pure +
    generic: it never interprets a key or payload.
    """
    groups: OrderedDict[Hashable, OrderedDict[Hashable, Any]] = OrderedDict()
    for rec in records:
        gk = reducer.group_key(rec)
        ik = reducer.instance_key(rec)
        bucket = groups.setdefault(gk, OrderedDict())
        if ik in bucket:
            raise ValueError(
                f"fan-out merge: duplicate instance {ik!r} in group {gk!r} "
                f"— a unit was written more than once across shards."
            )
        bucket[ik] = reducer.payload(rec)

    reduced: list[Hashable] = []
    for gk, bucket in groups.items():
        expected = set(reducer.expected_instances(gk))
        got = set(bucket)
        if got != expected:
            missing = sorted(expected - got, key=repr)
            extra = sorted(got - expected, key=repr)
            raise ValueError(
                f"fan-out merge: group {gk!r} INCOMPLETE — "
                f"missing={missing} extra={extra} "
                f"(expected {sorted(expected, key=repr)}, "
                f"got {sorted(got, key=repr)}). Refusing to aggregate a partial "
                f"series — short-count guard (RED-RULING-WORKSTREAM-B §2)."
            )
        series = [(ik, bucket[ik]) for ik in sorted(bucket, key=repr)]
        reducer.reduce(gk, series, out_root)
        reduced.append(gk)
    return reduced


# ── generic: manifest concat (PURE json union; offline-tested) ──────────────

def _status_priority(status: str) -> int:
    # completed wins over failed wins over pending, when ranks disagree on a
    # task id (round-robin makes disjoint task sets the norm; this only matters
    # if a task somehow appears in >1 rank manifest).
    return {"pending": 0, "failed": 1, "completed": 2}.get(status, 0)


def concat_manifests(rank_manifest_paths: Sequence[str], out_path: str) -> dict:
    """Union per-rank campaign manifests at the JSON level (schema =
    CampaignManifest: a ``tasks`` dict of ``task_id -> status``). A task is
    completed in the merged manifest iff completed on any rank; otherwise the
    highest-priority status seen is kept. ``total_tasks`` is recomputed. Pure
    stdlib (json), so it stays generic and offline-checkable and does not couple
    the merge to the manifest class internals."""
    merged_tasks: dict[str, str] = {}
    base: dict[str, Any] | None = None
    for p in rank_manifest_paths:
        with open(p, encoding="utf-8") as f:
            m = json.load(f)
        if base is None:
            base = dict(m)
        for tid, status in m.get("tasks", {}).items():
            prev = merged_tasks.get(tid)
            if prev is None or _status_priority(status) > _status_priority(prev):
                merged_tasks[tid] = status
    if base is None:
        base = {}
    base["tasks"] = merged_tasks
    base["total_tasks"] = len(merged_tasks)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)
    return base


# ── generic: HDF5 union (lazy h5py; LUMI-exercised by the byte-identity gate) ─

def union_hdf5(rank_h5_paths: Sequence[str], out_h5_path: str) -> int:
    """Copy every leaf group from each per-rank HDF5 into one output file. The
    per-unit groups are disjoint by construction (round-robin shards distinct
    units, each writing a distinct ``/byo/.../seed_NNNN/placements/.../env``
    path), so this is a straight copy-in with no rewrite — shape-agnostic, valid
    for any per-unit-group sweep. Fails loud on a path collision, which would
    signal a sharding bug (two ranks claiming the same unit). Returns the number
    of groups copied."""
    import h5py  # lazy: only the merge step needs it

    copied = 0
    with h5py.File(out_h5_path, "w") as dst:
        for src_path in rank_h5_paths:
            with h5py.File(src_path, "r") as src:
                def _copy(name, obj):
                    nonlocal copied
                    if isinstance(obj, h5py.Dataset):
                        return
                    # copy only leaf groups that hold datasets (the unit groups)
                    if not any(isinstance(v, h5py.Dataset) for v in obj.values()):
                        return
                    if name in dst:
                        raise ValueError(
                            f"fan-out merge: HDF5 group {name!r} present in more "
                            f"than one rank shard — sharding bug (a unit was "
                            f"assigned to >1 rank)."
                        )
                    parent = os.path.dirname(name)
                    if parent:
                        dst.require_group(parent)  # idempotent for shared parents
                    src.copy(obj, dst, name=name)
                    copied += 1
                src.visititems(_copy)
    return copied


# ── generic: vacuous-merge guard (PURE; offline-tested) ─────────────────────

def _assert_union_nonvacuous(
    n_groups: int, n_records: int, reducer_name: str
) -> None:
    """Fail loud on a vacuous merge BEFORE the completeness assertion can pass
    over an empty set (RED-RULING-MERGE-CLI-FOLLOWUP §3(c)). Two cases:

      - the union produced NO leaf groups (empty/missing shards) — the CLI's
        ``if not rank_h5s`` catches missing *files*, not present-but-empty ones;
      - the union produced groups but the selected reducer extracted ZERO records
        (a wrong-reducer mis-route, or total data loss). A battery file fed to
        the BYO reducer matches no ``autocorrelator`` sentinel -> 0 records -> the
        completeness loop never runs -> a silent exit-0 vacuous pass. This guard
        makes that loud.

    SCOPE: TOTAL loss / mis-route only. It does NOT cover a PARTIAL wholly-absent
    group (some groups present, one gone) — extract still returns >0 records
    there; that case is closed only by the expected-group inventory
    (RED-RULING-MERGE-CLI-FOLLOWUP ask-2 option (i)).
    """
    if n_groups == 0:
        raise ValueError(
            "fan-out merge: the union produced 0 leaf groups — empty or missing "
            "rank shards (nothing to merge)."
        )
    if n_records == 0:
        raise ValueError(
            f"fan-out merge: unioned {n_groups} leaf group(s) but reducer "
            f"{reducer_name} extracted 0 records — wrong reducer for this data "
            f"(reducer-selection mis-route) or total data loss. Refusing a "
            f"vacuous exit-0 merge (RED-RULING-MERGE-CLI-FOLLOWUP §3(c))."
        )


# ── orchestrator ────────────────────────────────────────────────────────────

def merge_shards(
    rank_h5_paths: Sequence[str],
    out_h5_path: str,
    reducer: ShardReducer,
    out_root: str,
    rank_manifest_paths: Sequence[str] | None = None,
    out_manifest_path: str | None = None,
) -> list[Hashable]:
    """Full merge: union the per-rank HDF5, extract records via the reducer,
    assert completeness + reduce per group, and (optionally) concat manifests.
    Returns the reduced group keys."""
    n_groups = union_hdf5(rank_h5_paths, out_h5_path)
    records = list(reducer.extract(out_h5_path))
    _assert_union_nonvacuous(n_groups, len(records), type(reducer).__name__)
    reduced = assert_complete_and_reduce(records, reducer, out_root)
    if rank_manifest_paths and out_manifest_path:
        concat_manifests(rank_manifest_paths, out_manifest_path)
    return reduced


# ── the BYO floquet reducer (the pluggable per-path piece) ──────────────────

def byo_dat_subpath(group_key: Hashable) -> str:
    """The byo_dat subdirectory (relative to out_root) for a BYO group key
    ``(stem, phys, env, obs_tail)``, reproducing the single-node engine layout
    EXACTLY. The engine builds
    ``os.path.join(out_root, stem, phys, env) + byo_observable_subpath(...)``;
    ``obs_tail`` is that observable subpath read back from the written HDF5 path,
    so the merged and single-node layouts cannot drift."""
    stem, phys, env, obs_tail = group_key
    p = os.path.join(stem, phys, env)
    if obs_tail:
        p = os.path.join(p, obs_tail)
    return p


class ByoAutocorrReducer:
    """BYO/floquet reducer: per-unit HDF5 groups are
    ``/byo/{stem}/seeds/seed_{NNNN}/placements/{phys}/{env}{obs}`` with an
    ``autocorrelator`` dataset; the aggregation group is
    ``(stem, phys, env, obs_tail)`` and the instance is the seed. ``reduce`` is
    the certified ``aggregate_byo_autocorr`` (relocated to merge-time, not
    reimplemented) over the complete per-group seed series. The series order is
    immaterial to the output (the mean/sem are over axis 0 and each per-instance
    .dat is named by its seed), so the merged .dat is byte-identical regardless
    of gather order.

    ``expected_seeds`` is the run's seed_list — the complete instance set every
    group must have; completeness is asserted against it before aggregation.
    ``out_subpath_fn`` maps a group key to its byo_dat subdirectory; it defaults
    to ``byo_dat_subpath`` (the single-node layout) and is injectable for tests
    or alternate layouts."""

    def __init__(
        self,
        expected_seeds: Iterable[int],
        out_subpath_fn: Callable[[Hashable], str] | None = None,
    ) -> None:
        self._expected = set(int(s) for s in expected_seeds)
        self._out_subpath_fn = out_subpath_fn or byo_dat_subpath

    def extract(self, merged_h5_path: str) -> Iterable[dict]:
        import h5py

        out: list[dict] = []
        with h5py.File(merged_h5_path, "r") as f:
            def _visit(name, obj):
                if not isinstance(obj, h5py.Dataset) or not name.endswith(
                    "autocorrelator"
                ):
                    return
                grp = obj.parent
                # path: byo/{stem}/seeds/seed_NNNN/placements/{phys}/{env}[/obs...]
                parts = name.split("/")
                try:
                    si = parts.index("seeds")
                    pi = parts.index("placements")
                except ValueError:
                    return
                # The RESOLVED script stem the writer used is the segment before
                # "seeds" (byo/{stem}/seeds/...). It already encodes whatever
                # family-collision disambiguation the single-node path applied
                # (that disambiguation lives in obs_tail, also read from the
                # written path), so carrying the stem here reproduces the
                # single-node byo_dat layout {stem}/{phys}/{env}{obs} EXACTLY.
                # Dropping it (pre-RED-REVIEW-WORKSTREAM-B-INCREMENTS-1-2) left
                # out_subpath_fn unable to rebuild the directory; the gate's
                # path-set-equality assert proves the carried stem is correct.
                stem = parts[si - 1]
                seed = int(parts[si + 1].split("_")[1])
                phys = parts[pi + 1]
                env = parts[pi + 2]
                obs_tail = "/".join(parts[pi + 3 : -1])  # observable/family levels
                out.append(
                    {
                        "group": (stem, phys, env, obs_tail),
                        "seed": seed,
                        "vector": list(obj[()]),
                    }
                )
            f.visititems(_visit)
        return out

    def group_key(self, record: dict) -> Hashable:
        return record["group"]

    def instance_key(self, record: dict) -> Hashable:
        return record["seed"]

    def payload(self, record: dict) -> Any:
        return record["vector"]

    def expected_instances(self, group_key: Hashable) -> Iterable[Hashable]:
        return self._expected

    def reduce(
        self, group_key: Hashable, series: list[tuple[Hashable, Any]], out_root: str
    ) -> None:
        from lumi_hpc_qc.sweep.byo_observable import aggregate_byo_autocorr

        sub = os.path.join(out_root, self._out_subpath_fn(group_key))
        aggregate_byo_autocorr(series, sub)

# ── the synthetic-channel battery reducer (the pluggable per-path piece) ─────

class BatteryReducer:
    """Hamiltonian twin-battery reducer (the non-BYO genericity proof). Per-unit
    HDF5 groups are
    ``devices/{device_prefix}/seeds/seed_{NNNN}/placements/{device_prefix}-{qubits}/calibrations/{cal_id}/{noise_config}[/params_{hash}]``,
    each holding an ``energy_trajectory`` dataset (one final twin result). The
    aggregation group is ``(device_prefix, placement, cal_id, noise_config,
    params_tail)`` and the instance is the seed.

    ``reduce`` is a NO-OP. The synthetic-channel battery writes each
    ``(seed, placement, env)`` twin result as a final, self-contained HDF5
    record — there is NO cross-instance aggregation in the battery path
    (verified against the tree: no ``def aggregate_*`` for it, no post-loop
    reduction in ``_execute_group``). ``union_hdf5`` already produced the per-unit
    groups in the merged HDF5, so there is nothing for ``reduce`` to relocate.

    The generic driver is still used for its COMPLETENESS assertion, which is the
    value here: a lost rank shard would silently drop a node's units, and grouping
    by ``(device, placement, cal, noise)`` + asserting the full ``seed_list`` per
    group catches exactly that. So for the battery path the completeness guard
    does everything and ``reduce`` does nothing — the cleanest demonstration that
    the generic shard+union+completeness core is reducer-independent
    (RED-RULING-ITEM3 §1/§2: an identity-reduce path satisfies the §4 genericity
    requirement). There is no ``.dat`` tree, so no ``out_subpath_fn``.

    ``params_tail`` is the battery analog of ``ByoAutocorrReducer``'s ``obs_tail``:
    the LHS-mode ``params_{hash}`` sublevel (empty in grid mode), carried so units
    that share ``(device, placement, cal, noise)`` but differ in hamiltonian
    parameters do not collapse into one group. Grid-mode units have none, so the
    group key reduces to exactly the approved 4-tuple.

    ``expected_seeds`` is the run's seed_list — the complete instance set every
    group must have; completeness is asserted against it.
    """

    def __init__(self, expected_seeds: Iterable[int]) -> None:
        self._expected = set(int(s) for s in expected_seeds)

    def extract(self, merged_h5_path: str) -> Iterable[dict]:
        import h5py

        out: list[dict] = []
        with h5py.File(merged_h5_path, "r") as f:
            def _visit(name, obj):
                # sentinel: the per-unit energy_trajectory dataset (every battery
                # unit writes exactly one) — the battery analog of BYO's
                # autocorrelator. Yields one record per unit.
                if not isinstance(obj, h5py.Dataset) or not name.endswith(
                    "energy_trajectory"
                ):
                    return
                # path: devices/{prefix}/seeds/seed_NNNN/placements/{placement}/
                #       calibrations/{cal_id}/{noise}[/params_HASH]/energy_trajectory
                parts = name.split("/")
                try:
                    di = parts.index("devices")
                    si = parts.index("seeds")
                    pi = parts.index("placements")
                    ci = parts.index("calibrations")
                except ValueError:
                    return
                device_prefix = parts[di + 1]
                seed = int(parts[si + 1].split("_")[1])
                placement = parts[pi + 1]
                cal_id = parts[ci + 1]
                noise_config = parts[ci + 2]
                # LHS params sublevel (parts[ci+3 : -1]); empty in grid mode.
                params_tail = "/".join(parts[ci + 3 : -1])
                out.append(
                    {
                        "group": (
                            device_prefix, placement, cal_id,
                            noise_config, params_tail,
                        ),
                        "seed": seed,
                    }
                )
            f.visititems(_visit)
        return out

    def group_key(self, record: dict) -> Hashable:
        return record["group"]

    def instance_key(self, record: dict) -> Hashable:
        return record["seed"]

    def payload(self, record: dict) -> Any:
        return None  # identity reduce consumes no payload

    def expected_instances(self, group_key: Hashable) -> Iterable[Hashable]:
        return self._expected

    def reduce(
        self, group_key: Hashable, series: list[tuple[Hashable, Any]], out_root: str
    ) -> None:
        # No-op: the battery's per-unit HDF5 records (already unioned) are the
        # final output; there is no cross-instance aggregation to relocate. The
        # completeness assertion in assert_complete_and_reduce is the value here.
        return


# ── reducer selection by experiment type (PURE mapping; offline-tested) ─────

_BYO_TYPES = frozenset({"byo_circuit"})
_BATTERY_TYPES = frozenset({"characterization", "vqe_sweep"})


def select_reducer(
    experiment_types: Iterable[str] | None,
    explicit_type: str | None,
    expected_seeds: Iterable[int],
):
    """Pick the merge reducer from the sweep's experiment type(s)
    (RED-RULING-MERGE-CLI-FOLLOWUP §3(a)/(b)). The production merge CLI MUST NOT
    hardcode a reducer: a battery file fed to the BYO reducer silently
    vacuous-passes (§3(c)). Mapping, all fail-loud:

      - byo_circuit                  -> ByoAutocorrReducer (aggregating)
      - characterization | vqe_sweep -> BatteryReducer     (identity reduce)
      - anything else                -> raise (a new, possibly AGGREGATING type
        must force a deliberate reducer choice; never fall through to
        identity-reduce — that is the silent-wrong-merge class)

    Precedence: an explicit type (the CLI's --experiment-type override) wins over
    the config-derived set. Neither given -> raise (no silent default). A config
    with MORE THAN ONE distinct type -> raise: a single-reducer merge cannot
    carry a mixed sweep, and running one reducer would vacuous-pass the other
    path's completeness guard.

    (Allowlist note: the only experiment types in the tree are byo_circuit and
    characterization; vqe_sweep is engine-recognized, non-aggregating, with no
    example config yet — forward-looking. random_regular is a graph_type under
    model_params, NOT an experiment type, so it is correctly absent here.)
    """
    if explicit_type is not None:
        resolved = {explicit_type}
    elif experiment_types is not None:
        resolved = {str(t) for t in experiment_types}
    else:
        resolved = set()

    if not resolved:
        raise ValueError(
            "no experiment type for reducer selection — provide --config (to "
            "read the sweep type) or --experiment-type."
        )
    if len(resolved) > 1:
        raise ValueError(
            f"mixed experiment types {sorted(resolved)} in one config — a "
            f"single-reducer merge cannot carry a mixed sweep (running one "
            f"reducer would vacuous-pass the other path's completeness guard). "
            f"Split the campaign by type, or merge per-type."
        )
    (exp_type,) = tuple(resolved)
    if exp_type in _BYO_TYPES:
        return ByoAutocorrReducer(expected_seeds=expected_seeds)
    if exp_type in _BATTERY_TYPES:
        return BatteryReducer(expected_seeds=expected_seeds)
    raise ValueError(
        f"unknown experiment type {exp_type!r} for reducer selection — known: "
        f"{sorted(_BYO_TYPES | _BATTERY_TYPES)}. A new type must add an explicit "
        f"reducer mapping (an aggregating type must NEVER fall through to "
        f"identity-reduce)."
    )
