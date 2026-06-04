"""Patch C proof — noiseless placement-dedup + broadcast/re-stamp.

RED-RULING-BYO-FLAT-DISPATCH-AND-NOISELESS-DEDUP §2(C). Pure-logic replicas of
the two flat-dispatch dedup rules in _execute_byo_group; the real .dat byte-
identity (dedup ON vs OFF, grid + LHS) is the §5.4 LUMI gate.

The standing physics fact (verified in prepare.py + the worker): the noiseless arm
is AerSimulator transpiled as-written -> placement_phys_qubits is IGNORED, so the
noiseless payload for a given (family, seed) is IDENTICAL across placements. This
test ASSUMES that (the LUMI gate confirms it empirically) and proves the plumbing:

1. Build dedup-skip: with the flag on, noiseless units are built ONCE on the
   canonical placement (P x S -> S per family); device-cal stays P x S. Off ->
   both P x S (byte-identical to pre-C).
2. Merge broadcast/re-stamp: the deduped noiseless result, broadcast to every
   placement with placement_id + physical_qubit_set re-stamped, reproduces the
   EXACT record set (group key + payload) a non-deduped run would have written —
   the .dat aggregation groups by (physical_qubit_set, env, observable,
   circuit_function), so the group set and per-group content must match.

Self-running; also pytest-collectable.
"""


class Placement:
    def __init__(self, pid, phys):
        self.placement_id = pid
        self.qubit_mapping = {i: q for i, q in enumerate(phys)}

    def phys(self):
        return [self.qubit_mapping[i] for i in range(len(self.qubit_mapping))]


PLACEMENTS = [Placement("p0", ["q1", "q2"]),
              Placement("p1", ["q5", "q6"]),
              Placement("p2", ["q9", "q10"])]
SEEDS = [0, 1]
ENVS = ["device_calibrated", "noiseless"]
FAMILY = ("autocorr", "build_circuit")


def payload(env, seed, placement_id):
    # Physics fact: noiseless is placement-INDEPENDENT; device-cal is not.
    if env == "noiseless":
        return ("auto", seed)              # no placement term
    return ("auto", seed, placement_id)    # placement-dependent


def build_units(placements, dedup):
    """Mirror of the helper build loop's env handling."""
    canonical = placements[0].placement_id
    units = []
    for p in placements:
        for seed in SEEDS:
            for env in ENVS:
                if dedup and env == "noiseless" and p.placement_id != canonical:
                    continue
                units.append((p, seed, env))
    return units


def merge_records(units, placements, dedup):
    """Mirror of the merge: one record per unit, broadcasting deduped noiseless."""
    recs = []
    for (p, seed, env) in units:
        base = dict(placement_id=p.placement_id, physical_qubit_set=tuple(p.phys()),
                    env=env, observable=FAMILY[0], circuit_function=FAMILY[1],
                    seed=seed, payload=payload(env, seed, p.placement_id))
        if dedup and env == "noiseless":
            for bp in placements:
                rec = dict(base)
                rec["placement_id"] = bp.placement_id
                rec["physical_qubit_set"] = tuple(bp.phys())
                recs.append(rec)
        else:
            recs.append(base)
    return recs


def group_keyset(recs):
    return {(r["physical_qubit_set"], r["env"], r["observable"],
             r["circuit_function"], r["seed"]) for r in recs}


def canon(recs):
    return sorted((r["placement_id"], r["physical_qubit_set"], r["env"],
                   r["observable"], r["circuit_function"], r["seed"],
                   r["payload"]) for r in recs)


def test_dedup_off_builds_per_placement():
    u = build_units(PLACEMENTS, dedup=False)
    nl = [x for x in u if x[2] == "noiseless"]
    dc = [x for x in u if x[2] == "device_calibrated"]
    assert len(nl) == len(PLACEMENTS) * len(SEEDS)
    assert len(dc) == len(PLACEMENTS) * len(SEEDS)


def test_dedup_on_builds_noiseless_once():
    u = build_units(PLACEMENTS, dedup=True)
    nl = [x for x in u if x[2] == "noiseless"]
    dc = [x for x in u if x[2] == "device_calibrated"]
    assert len(nl) == len(SEEDS), "noiseless computed once per seed on canonical placement"
    assert len(dc) == len(PLACEMENTS) * len(SEEDS), "device-cal unchanged"
    assert all(p.placement_id == PLACEMENTS[0].placement_id for (p, _, e) in u if e == "noiseless")


def test_broadcast_reproduces_nondeduped_record_set():
    ref = merge_records(build_units(PLACEMENTS, dedup=False), PLACEMENTS, dedup=False)
    ded = merge_records(build_units(PLACEMENTS, dedup=True), PLACEMENTS, dedup=True)
    assert canon(ref) == canon(ded), "deduped+broadcast record set must equal non-deduped"


def test_broadcast_group_keyset_matches():
    ref = merge_records(build_units(PLACEMENTS, dedup=False), PLACEMENTS, dedup=False)
    ded = merge_records(build_units(PLACEMENTS, dedup=True), PLACEMENTS, dedup=True)
    assert group_keyset(ref) == group_keyset(ded)
    nl_keys = {k for k in group_keyset(ded) if k[1] == "noiseless"}
    assert len(nl_keys) == len(PLACEMENTS) * len(SEEDS), "P x S distinct noiseless .dat groups"


def test_restamp_targets_only_placement_fields():
    ded = merge_records(build_units(PLACEMENTS, dedup=True), PLACEMENTS, dedup=True)
    nl = [r for r in ded if r["env"] == "noiseless"]
    for r in nl:
        # physical_qubit_set must correspond to its stamped placement_id
        pid_to_phys = {p.placement_id: tuple(p.phys()) for p in PLACEMENTS}
        assert r["physical_qubit_set"] == pid_to_phys[r["placement_id"]]
        # payload stays the placement-independent noiseless value
        assert r["payload"] == ("auto", r["seed"])


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nALL {len(fns)} checks PASS")
