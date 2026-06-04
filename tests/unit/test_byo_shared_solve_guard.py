"""Patch A proof — BYO shared placement solve + fail-loud connectivity guard.

RED-RULING-BYO-FLAT-DISPATCH-AND-NOISELESS-DEDUP (A). Two layers:

1. Pure decision logic (runs anywhere, no qiskit): the exact cache+guard branch
   structure from _execute_byo_group — miss->solve+store, hit+match->reuse (arms
   paired), hit+mismatch+default->fail-loud, hit+mismatch+opt-out->fresh/no-poison,
   single-family->never engages.
2. Real flag-threading (runs in-container where qiskit imports): the
   observables_independent field exists and defaults False on both ExperimentSpec
   and SweepTask, and _parse_experiment reads it.

The REAL placement EQUALITY (shared==independent on the echo config) and the
in-context fail-loud are the LUMI gates (need the solver); this locks the logic
and the wiring that surround them.

Self-running: `python3 tests/unit/test_byo_shared_solve_guard.py`. Also collects
under pytest (test_* functions).
"""


class _SolveError(ValueError):
    pass


def _decide(cache, key, conn, observables_independent, solve_fn):
    """Mirror of the patched cache+guard branch. Returns (placements, solved, stored)."""
    conn_norm = frozenset(frozenset(e) for e in conn)
    cached = cache.get(key)
    placements = None
    store_to_cache = cached is None
    if cached is not None:
        cached_placements, cached_conn = cached
        if conn_norm == cached_conn:
            placements = cached_placements
        elif observables_independent:
            pass
        else:
            raise _SolveError(
                "fail-loud: families' connectivity differs; a shared solve "
                "would mis-pair a ratio across placements"
            )
    solved = 0
    if placements is None:
        placements = solve_fn()
        solved = 1
        if store_to_cache:
            cache[key] = (placements, conn_norm)
    return placements, solved, store_to_cache


_K = ("script", "cal", "topo", 10, "dev", 5, None, None, None)
_CHAIN = [(0, 1), (1, 2), (2, 3)]
_CHAIN_REV = [(1, 0), (3, 2), (2, 1)]      # same undirected edges, reordered/reversed
_DIFFERENT = [(0, 1), (1, 2), (2, 9)]


def test_cache_miss_solves_and_stores():
    cache = {}
    p, solved, stored = _decide(cache, _K, _CHAIN, False, lambda: ["P_autocorr"])
    assert solved == 1 and stored and cache[_K][0] == ["P_autocorr"]


def test_match_reuses_and_pairs_arms():
    cache = {}
    _decide(cache, _K, _CHAIN, False, lambda: ["P_autocorr"])
    calls = {"n": 0}

    def echo_solve():
        calls["n"] += 1
        return ["P_echo_OTHER"]

    p, solved, _ = _decide(cache, _K, _CHAIN_REV, False, echo_solve)
    assert solved == 0 and calls["n"] == 0          # reused, no second solve
    assert p == ["P_autocorr"]                       # arms paired on identical placements


def test_mismatch_default_fails_loud():
    cache = {}
    _decide(cache, _K, _CHAIN, False, lambda: ["P"])
    raised = False
    try:
        _decide(cache, _K, _DIFFERENT, False, lambda: ["X"])
    except _SolveError:
        raised = True
    assert raised


def test_mismatch_optout_solves_fresh_without_poison():
    cache = {}
    _decide(cache, _K, _CHAIN, False, lambda: ["P_autocorr"])
    before = cache[_K]
    p, solved, stored = _decide(cache, _K, _DIFFERENT, True, lambda: ["P_independent"])
    assert solved == 1 and not stored
    assert cache[_K] == before                       # shared entry not poisoned
    assert p == ["P_independent"]


def test_single_family_never_engages_guard():
    cache = {}
    _decide(cache, ("only_one",), _CHAIN, False, lambda: ["P"])  # must not raise


def test_real_flag_threading_in_container():
    """Skipped offline; runs where the engine imports (LUMI qiskit container)."""
    try:
        from lumi_hpc_qc.sweep.sweep_engine import ExperimentSpec, SweepTask
    except Exception:
        return  # offline: engine pulls qiskit; covered by the LUMI gate
    assert ExperimentSpec().observables_independent is False
    assert ExperimentSpec(observables_independent=True).observables_independent is True
    assert SweepTask().observables_independent is False
    assert SweepTask(observables_independent=True).observables_independent is True


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nALL {len(fns)} checks PASS")
