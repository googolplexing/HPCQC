"""Patch B proof — flat cross-observable dispatch must stratify by (env, observable).

RED-RULING-BYO-FLAT-DISPATCH-AND-NOISELESS-DEDUP §3(B). Executed against the REAL
fanout.shard_indices (loaded by path offline; normal import in-container).

The multi-seed-BYO wholly-absent-group protection rides on the shard spreading
each aggregation group's seeds across ranks, so a dropped rank SHORT-COUNTS the
group (the short-count guard fires) rather than losing it whole (BYO has no
inventory — a wholly-absent group is a silent lost shard). Adding the observable
axis with strata=env breaks that: with the observable as the innermost build axis
(units alternate autocorr/echo), a 2-rank round-robin within the single env
stratum sends ALL of one family to one rank — drop it and whole (placement, env,
observable) groups vanish. strata=(env, observable) round-robins within each
family's stratum, so every group's seeds spread again.

Asserts: strata=env -> at least one group WHOLLY absent on a rank drop (exposed);
strata=(env, observable) -> NO group wholly absent on any single-rank drop, and
each group is short-counted (guard-visible). Self-running; also pytest-collectable.
"""

import importlib.util
import os


def _load_shard_indices():
    try:
        from lumi_hpc_qc.sweep.fanout import shard_indices  # in-container
        return shard_indices
    except Exception:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, "..", "..", "src", "lumi_hpc_qc", "sweep", "fanout.py")
        spec = importlib.util.spec_from_file_location("_fanout_standalone", os.path.abspath(path))
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        return m.shard_indices


_shard_indices = _load_shard_indices()

# Build order mirrors the flat dispatch: placement -> seed -> env -> observable
# (observable innermost, so consecutive units alternate families — the layout
# that exposed strata=env in Red's execution).
PLACEMENTS = [0, 1]
SEEDS = [0, 1]
ENVS = ["device_calibrated"]
OBSERVABLES = ["autocorr", "echo"]

UNITS = [
    {"placement": p, "env": e, "obs": o, "seed": s}
    for p in PLACEMENTS
    for s in SEEDS
    for e in ENVS
    for o in OBSERVABLES
]

# Aggregation group = (placement, env, observable); members are its seeds.
ALL_GROUPS = {(u["placement"], u["env"], u["obs"]) for u in UNITS}
GROUP_SIZE = {g: sum(1 for u in UNITS if (u["placement"], u["env"], u["obs"]) == g)
              for g in ALL_GROUPS}


def _groups_on_rank(strata_fn, rank, nranks):
    strata = [strata_fn(u) for u in UNITS]
    idxs = _shard_indices(len(UNITS), rank, nranks, strata)
    kept = [UNITS[i] for i in idxs]
    counts = {}
    for u in kept:
        g = (u["placement"], u["env"], u["obs"])
        counts[g] = counts.get(g, 0) + 1
    return counts


_ENV = lambda u: u["env"]
_ENV_OBS = lambda u: (u["env"], u["obs"])


def test_strata_env_wholly_absents_a_group_on_rank_drop():
    """The bug: strata=env loses whole (placement,env,obs) groups when a rank drops."""
    nranks = 2
    exposed = False
    for dropped in range(nranks):
        kept = _groups_on_rank(_ENV, 1 - dropped, nranks)  # the surviving rank
        if set(kept.keys()) != ALL_GROUPS:
            exposed = True
    assert exposed, "expected strata=env to wholly-absent a group on some rank drop"


def test_strata_env_obs_protects_no_group_wholly_absent():
    """The fix: strata=(env,observable) keeps every group present (short-counted)."""
    nranks = 2
    for dropped in range(nranks):
        kept = _groups_on_rank(_ENV_OBS, 1 - dropped, nranks)
        assert set(kept.keys()) == ALL_GROUPS, (
            f"group wholly absent under (env,obs) when rank {dropped} dropped: "
            f"missing {ALL_GROUPS - set(kept.keys())}"
        )
        for g, full in GROUP_SIZE.items():
            assert kept[g] < full, (
                f"group {g} not short-counted (got {kept[g]} of {full}) — the "
                "short-count guard needs a deficit to fire"
            )


def test_both_ranks_together_lose_nothing():
    """Sanity: union of all ranks reconstructs every unit exactly once (no loss/dup)."""
    nranks = 2
    seen = []
    for r in range(nranks):
        strata = [_ENV_OBS(u) for u in UNITS]
        seen += _shard_indices(len(UNITS), r, nranks, strata)
    assert sorted(seen) == list(range(len(UNITS))), "shard must partition units exactly"


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nALL {len(fns)} checks PASS (real fanout.shard_indices)")
