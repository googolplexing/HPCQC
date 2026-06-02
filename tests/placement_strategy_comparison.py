#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Placement-strategy comparison — how different ways of choosing physical qubits
for a circuit of a given SHAPE compare on one calibration.

DIAGNOSTIC (not a CI gate). It runs the placement solver several ways on the same
device graph and prints a side-by-side report so a researcher can see — and
understand WHY — each strategy picks the qubits it does, what that means for
device coverage and for parallel execution on the QPU, and exactly what to put in
a sweep YAML to reproduce each.

Sections:
  §1 FIDELITY TOP-N   — the N highest-fidelity placements (current default).
                        Optimises per-placement quality; tends to CLUSTER.
  §2 DISJOINT         — N mutually-independent placements (no shared qubit).
                        Optimises device COVERAGE; count=auto = the device-max.
  §3 PARALLEL ROUNDS  — pack a chosen set into execution rounds where every chain
                        in a round can run SIMULTANEOUSLY (the QPU schedule view;
                        DSatur + greedy). Packing is QPU-scheduling, NOT a sim
                        YAML knob — in simulation each chain fans across CPUs
                        independently.
  §4 YAML PERMUTATIONS — the valid placement-specification patterns, each emitted
                        as a paste-ready sweep experiment block:
                          A manual only           (physical_qubits)
                          B solver top-N          (placement: top_N)
                          C manual + solver next-N (physical_qubits + placement)
                          D diversity, disjoint   (placement_diversity)
                          INVALID: manual + diversity (rejected at parse)

Needs only the solver (rustworkx + cal adapter) — NO aer/h5py, no sweep.

Usage (positional; campaign defaults):
    python3 tests/placement_strategy_comparison.py [CAL] [N_PLACEMENTS] [CHAIN_QUBITS] [SHAPE]
    SHAPE in: chain (default) | ring | star | grid | ladder | complete
Or:  sbatch tests/slurm_placement_comparison.sh [CAL] [N] [QUBITS] [SHAPE]

Forward-compatible with the pending max_overlap removal: disjoint calls pass only
`count=`, relying on the strict-disjoint default.
"""

from __future__ import annotations

import math
import os
import sys
from collections import deque

project_dir = os.environ.get(
    "PROJECT_DIR",
    os.environ.get(
        "SINGULARITYENV_PROJECT_DIR",
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ),
)
sys.path.insert(0, os.path.join(project_dir, "src"))

DEFAULT_CAL = os.path.join(
    project_dir, "examples", "q50_calibration_20260524_08c3c70f.json"
)


# ── circuit shapes ──────────────────────────────────────────────────────────

def make_shape(shape: str, n: int) -> list[tuple[int, int]]:
    """Circuit connectivity (edges) for a named shape of n qubits.

    Raises ValueError on (shape, n) combinations that are ill-defined (e.g. a
    grid whose n doesn't factor). Whether the shape can EMBED on the device is a
    separate question answered by the solver (0 placements) + explain_empty()."""
    s = shape.lower()
    if s in ("chain", "line", "path"):
        return [(i, i + 1) for i in range(n - 1)]
    if s in ("ring", "cycle"):
        return [(i, (i + 1) % n) for i in range(n)]
    if s == "star":
        return [(0, i) for i in range(1, n)]
    if s == "complete":
        return [(i, j) for i in range(n) for j in range(i + 1, n)]
    if s == "ladder":
        if n % 2:
            raise ValueError(f"ladder needs even n (two equal rails); got {n}")
        m = n // 2
        e = []
        for i in range(m - 1):
            e.append((i, i + 1))
            e.append((m + i, m + i + 1))
        for i in range(m):
            e.append((i, m + i))
        return e
    if s == "grid":
        r = math.isqrt(n)
        while r > 1 and n % r:
            r -= 1
        c = n // r
        if r * c != n or r < 2:
            raise ValueError(
                f"grid needs n to factor into a rectangle (r>=2); {n} doesn't"
            )
        e = []
        for a in range(r):
            for b in range(c):
                idx = a * c + b
                if b + 1 < c:
                    e.append((idx, idx + 1))
                if a + 1 < r:
                    e.append((idx, idx + c))
        return e
    raise ValueError(
        f"unknown shape {shape!r} (chain|ring|star|grid|ladder|complete)"
    )


def device_props(cal):
    """(bipartite, triangle_free, max_degree) of the device graph — the facts
    that decide which shapes can embed."""
    adj = cal.adjacency
    # bipartite (2-colourable) <=> no odd cycle
    color, bip = {}, True
    for s in adj:
        if s in color:
            continue
        color[s] = 0
        q = deque([s])
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in color:
                    color[v] = color[u] ^ 1
                    q.append(v)
                elif color[v] == color[u]:
                    bip = False
    tri = False
    for u in adj:
        nb = list(adj[u])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                if nb[j] in adj.get(nb[i], ()):
                    tri = True
    maxdeg = max((len(s) for s in adj.values()), default=0)
    return bip, (not tri), maxdeg


def explain_empty(shape: str, n: int, cal) -> str:
    """Why a shape yielded 0 placements, from the device topology facts."""
    bip, tri_free, maxdeg = device_props(cal)
    s = shape.lower()
    if s in ("ring", "cycle") and n % 2 == 1 and bip:
        return (f"a ring of {n} qubits needs a cycle of length {n} in the device "
                f"graph, but {cal.device_id}'s coupling graph is bipartite and has "
                f"only even-length cycles, so an odd-N ring has no embedding "
                f"(even N does).")
    if s == "star" and (n - 1) > maxdeg:
        return (f"a star with {n-1} leaves needs a qubit of degree {n-1}, but "
                f"{cal.device_id}'s maximum qubit degree is {maxdeg}, so a star "
                f"supports at most {maxdeg} leaves (N <= {maxdeg+1}).")
    if s == "complete" and n >= 3 and tri_free:
        return (f"a complete graph K{n} (N >= 3) contains triangles, but "
                f"{cal.device_id}'s coupling graph is triangle-free, so no all-to-"
                f"all circuit of 3+ qubits embeds; only K2 (a single coupled pair) "
                f"fits.")
    return (f"no placement exists: no subgraph of {cal.device_id}'s current "
            f"coupling graph matches this shape's required connectivity at "
            f"{n} qubits.")


# ── formatting helpers ──────────────────────────────────────────────────────

def hr(ch="─", n=78):
    return ch * n


def names(p, idx_to_name=None):
    """Qubit names in CIRCUIT (path) order, via the embedding's logical->physical
    map, so consecutive entries are coupled. This is required, not cosmetic: the
    manual physical_qubits loader validates positionally (placement_solver
    placements_from_names: circuit edge (a,b) must map to a calibrated coupler), so
    the emitted YAML and §4's resolve_placements call must use path order, not a
    sorted qubit set. Falls back to sorted physical_indices only if a mapping is
    somehow absent."""
    mapping = getattr(p, "qubit_mapping", None)
    if mapping:
        return [mapping[i] for i in sorted(mapping)]
    return [(idx_to_name or {}).get(i, f"QB{i}") for i in p.physical_indices]


def noise_aggregates(p):
    t2 = [q.get("t2_us", 0.0) for q in p.per_qubit_calibration.values()]
    return {
        "ro": p.avg_readout_fidelity,
        "cz": p.avg_gate_fidelity,
        "t2min": min(t2) if t2 else 0.0,
        "t2mean": (sum(t2) / len(t2)) if t2 else 0.0,
    }


def spread(placements):
    import itertools
    sets = [set(p.physical_indices) for p in placements]
    union = set().union(*sets) if sets else set()
    pair = [len(a & b) for a, b in itertools.combinations(sets, 2)]
    return (len(union), (sum(pair) / len(pair)) if pair else 0.0,
            max(pair) if pair else 0)


def print_table(placements, idx_to_name, limit=None):
    rows = placements if limit is None else placements[:limit]
    print(f"  {'#':>3}  {'score':>8}  {'ro_fid':>7}  {'cz_fid':>7}  "
          f"{'T2_min':>7}  {'T2_mean':>8}  qubits")
    for i, p in enumerate(rows, 1):
        a = noise_aggregates(p)
        qs = ", ".join(names(p, idx_to_name))
        print(f"  {i:>3}  {p.score:>8.5f}  {a['ro']:>7.4f}  {a['cz']:>7.4f}"
              f"  {a['t2min']:>7.1f}  {a['t2mean']:>8.1f}  [{qs}]")


# ── §4: paste-ready sweep experiment block (real campaign sweep-depth) ───────

def emit_experiment(label, placement_lines, cal_rel, qubits):
    print("    sweep:")
    print("      experiments:")
    print("        - type: byo_circuit")
    print(f"          label: {label}")
    print("          circuit_script: examples/byo/floquet_dtc_echo.py")
    print("          circuit_function: build_circuit")
    print(f"          fixed: {{num_qubits: {qubits}, epsilon: 0.03}}")
    print("          grid: {num_kicks: {range: [0, 40]}}            # sweep depth — adjust")
    print("          disorder: {source: file, file: examples/byo/floquet_disorder_q10_echo_ak10.json, initial_state: 3}  # q10 campaign disorder — adjust for other sizes")
    print("          disorder_gates: [rz, rzz]")
    for ln in placement_lines:
        print("          " + ln)
    print("          seed_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]      # sweep depth — adjust")
    print("          shots: 1000                                    # sweep depth — adjust")
    print("          noise_configs: [noiseless, device_calibrated]")
    print("      calibrations:")
    print(f"        - {cal_rel}")
    print()


# ── main ────────────────────────────────────────────────────────────────────

def main():
    from lumi_hpc_qc.sweep.placement_solver import (
        GeneralPlacementSolver,
        select_disjoint_placements,
    )
    from lumi_hpc_qc.plugins.calibration_adapters.iqm_v2 import IQMv2Adapter

    cal_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CAL
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    QUBITS = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    SHAPE = sys.argv[4] if len(sys.argv) > 4 else "chain"

    cal = IQMv2Adapter().load(cal_path)
    setid = getattr(cal, "calibration_set_id", "") or "?"
    cal_rel = "examples/" + os.path.basename(cal_path)
    solver = GeneralPlacementSolver()
    solver.add_device(cal)
    idx_to_name = cal.index_to_qubit_name

    try:
        circuit_edges = make_shape(SHAPE, QUBITS)
    except ValueError as e:
        print(f"ERROR building shape {SHAPE!r} of {QUBITS} qubits: {e}")
        return 2

    cands = solver.find_all_placements(
        circuit_edges=circuit_edges, circuit_qubits=QUBITS, strategy="max_fidelity",
    )

    print(hr("═"))
    print("  PLACEMENT STRATEGY COMPARISON")
    print(hr("═"))
    print(f"  device           : {cal.device_id}")
    print(f"  calibration      : {os.path.basename(cal_path)}")
    print(f"  calibration_set  : {setid}")
    print(f"  circuit          : {QUBITS}-qubit {SHAPE} ({len(circuit_edges)} edges)")
    print(f"  valid placements : {len(cands)} distinct qubit sets host this shape")
    print(f"  requested N      : {N} placements")
    print()

    if not cands:
        print(hr())
        print(f"  NO VALID PLACEMENTS for a {QUBITS}-qubit {SHAPE} on {cal.device_id}.")
        print(f"  WHY: {explain_empty(SHAPE, QUBITS, cal)}")
        print(hr())
        return 0

    print("  Each strategy picks from the SAME ranked list of valid embeddings;")
    print("  only the SELECTION RULE differs. Score = mean(readout, CZ) fidelity")
    print("  (higher better). T2 (us) is dephasing time; T2 contrast across regions")
    print("  is what makes a spatially-diverse sample richer than one good corner.")
    print()

    top_n = cands[:N]
    auto = select_disjoint_placements(cands, cal, count="auto")

    # ── §1 ─────────────────────────────────────────────────────────────────
    print(hr())
    print("  §1  FIDELITY TOP-N  (current default)")
    print(hr())
    print("  HOW CHOSEN: enumerate every valid placement, score by mean qubit+coupler")
    print(f"  fidelity, take the top {N}. Nothing stops two from reusing the same good")
    print("  qubits — and the best fidelity lives in one region, so they CLUSTER.")
    print()
    print_table(top_n, idx_to_name)
    d, mo, mx = spread(top_n)
    print()
    print(f"  COVERAGE: {len(top_n)} placements touch {d}/{cal.num_qubits} distinct "
          f"qubits; mean pairwise overlap {mo:.1f}/{QUBITS}, max {mx}/{QUBITS}.")
    print("  READING: high overlap = one region reshuffled, not a broad sample.")
    print()

    # ── §2 ─────────────────────────────────────────────────────────────────
    print(hr())
    print("  §2  DISJOINT  (spatially-independent placements)")
    print(hr())
    print("  HOW CHOSEN: walk the SAME ranked list, accept a placement only if it")
    print("  shares NO qubit with any already-accepted one. Each is the best still-")
    print("  available given what's taken. In simulation each one's device-calibrated")
    print("  noise is composed only from its own qubits, so they're independent by")
    print("  construction (the no-cross-talk property).")
    print()
    print(f"  count=auto -> {len(auto)} placement(s)  (the DEVICE-MAX: the most")
    print(f"   independent {QUBITS}q-{SHAPE} placements {cal.device_id} physically fits)")
    print_table(auto, idx_to_name)
    da, _, mxa = spread(auto)
    print()
    print(f"  COVERAGE: {len(auto)} placements touch {da}/{cal.num_qubits} distinct "
          f"qubits; max pairwise overlap {mxa} (independent).")
    print()
    print(f"  count={N} (requested) -> ", end="")
    try:
        sel_n = select_disjoint_placements(cands, cal, count=N)
        print(f"{len(sel_n)} placement(s).")
        print_table(sel_n, idx_to_name)
    except ValueError:
        print("CANNOT BE SATISFIED.")
        print(f"  WHY: {QUBITS}q x {N} = {QUBITS*N} qubits of independent chains, but")
        print(f"       only {len(auto)} fit on {cal.num_qubits} without sharing. The")
        print("       selector FAILS LOUD rather than quietly returning fewer (a short")
        print("       count masquerading as success is the failure class the guard")
        print(f"       catches). To run {N} placements, use disjoint ROUNDS — see §3.")
    print()
    print("  FULL NOISE-CHANNEL VALUES for the disjoint set (per qubit):")
    for ci, p in enumerate(auto):
        print(f"   c{ci}: {'-'.join(names(p, idx_to_name))}")
        print(f"     {'qubit':>6}  {'T1_us':>7}  {'T2_us':>7}  {'ro_fid':>7}  {'1q_err':>8}")
        for qn, q in sorted(p.per_qubit_calibration.items(),
                            key=lambda kv: int(kv[0][2:]) if kv[0][2:].isdigit() else 0):
            print(f"     {qn:>6}  {q.get('t1_us',0):>7.1f}  {q.get('t2_us',0):>7.1f}"
                  f"  {q.get('readout_fidelity',0):>7.4f}  {q.get('single_gate_error',0):>8.5f}")
    print()

    # ── §3 ─────────────────────────────────────────────────────────────────
    print(hr())
    print("  §3  PARALLEL ROUNDS  (how a chosen set schedules on the QPU)")
    print(hr())
    print("  HOW CHOSEN: pack a set into rounds where every chain in a round is")
    print("  mutually non-overlapping, so the round runs in one QPU shot. Fewer rounds")
    print("  = more parallelism. NOTE: this is the REAL-QPU schedule; in SIMULATION")
    print("  each chain is an independent job and fans across CPUs regardless (packing")
    print("  is not wired into the sim path). Two packers: DSatur (minimal) + greedy.")
    print()

    def show_rounds(label, placements):
        for strat, nm in (("optimal", "DSatur (minimal)"), ("greedy", "greedy")):
            rounds = solver.pack_rounds(placements, strategy=strat, packing_seed=42)
            print(f"  {label} via {nm}: {len(rounds)} round(s) for "
                  f"{len(placements)} placement(s)")
            for r in rounds[:6]:
                chains = "; ".join("-".join(names(p, idx_to_name)) for p in r.placements)
                print(f"     round {r.round_id}: {len(r.placements)} chain(s)  "
                      f"[{chains[:88]}{'...' if len(chains) > 88 else ''}]")
            if len(rounds) > 6:
                print(f"     ... ({len(rounds)-6} more rounds)")
            print()

    print("  (a) the FIDELITY TOP-N set:")
    show_rounds("top-N", top_n)
    print("  (b) the DISJOINT set (non-entangled circuits packed for parallel run):")
    show_rounds("disjoint", auto)
    waves = -(-N // max(1, len(auto)))
    print(f"  READING: clustered top-N can rarely co-run -> many serial rounds; the")
    print(f"  disjoint set is independent -> a SINGLE round (all parallel). To run")
    print(f"  N={N}, repeat disjoint rounds: ceil({N}/{len(auto)}) = {waves} waves of up")
    print(f"  to {len(auto)} non-entangled chains tile the device {N} placements deep.")
    print()

    # ── §4  YAML PERMUTATIONS ───────────────────────────────────────────────
    print(hr())
    print("  §4  YAML PERMUTATIONS  (valid placement specs; paste-ready, real")
    print("      campaign sweep-depth — adjust the commented fields for your run)")
    print(hr())
    print()
    manual = auto[-1] if auto else cands[0]      # an illustrative researcher pick
    manual_names = names(manual, idx_to_name)

    print("  A — MANUAL ONLY (solver bypassed; runs exactly your chains):")
    emit_experiment("manual_only", [
        "physical_qubits:",
        "  - [" + ", ".join(manual_names) + "]",
    ], cal_rel, QUBITS)

    print(f"  B — SOLVER TOP-N (solver picks the top {N} by fidelity):")
    print(f"  # resolves to the top {N} fidelity placements (see §1).")
    emit_experiment("solver_topN", [f"placement: top_{N}"], cal_rel, QUBITS)

    print("  C — MANUAL + SOLVER NEXT-3 (your pick PLUS the solver's next 3 best NEW):")
    union = solver.resolve_placements(
        circuit_edges=circuit_edges, circuit_qubits=QUBITS,
        device_id=cal.device_id, strategy="max_fidelity",
        manual_qubit_name_lists=[manual_names], solver_top_n=3,
    )
    print(f"  # resolves to {len(union)} placements (manual-first, then solver-ranked):")
    for p in union:
        print(f"  #   [{', '.join(names(p, idx_to_name))}]")
    emit_experiment("manual_plus_solver_top3", [
        "physical_qubits:",
        "  - [" + ", ".join(manual_names) + "]",
        "placement: top_3",
    ], cal_rel, QUBITS)

    print(f"  D — DIVERSITY, DISJOINT (solver picks the {len(auto)} independent regions):")
    print(f"  # resolves to the {len(auto)} disjoint chains shown in §2.")
    emit_experiment("diversity_disjoint",
                    ["placement_diversity: {strategy: disjoint, count: auto}"],
                    cal_rel, QUBITS)

    print("  INVALID — MANUAL + DIVERSITY (mutually exclusive; REJECTED AT PARSE):")
    print("  # you cannot ask the solver to diversify a set you pinned by hand.")
    print("        - type: byo_circuit")
    print("          physical_qubits:")
    print("            - [" + ", ".join(manual_names) + "]")
    print("          placement_diversity: {strategy: disjoint, count: auto}")
    print("          # ^ parse error: physical_qubits and placement_diversity are")
    print("          #   mutually exclusive. Use one or the other (A/D), not both.")
    print()

    print(hr("═"))
    print("  SUMMARY:")
    print(f"   * sample the DEVICE broadly  -> D (disjoint), {len(auto)} independent chains")
    print("   * probe the single BEST region -> A or B (fidelity top-N)")
    print("   * pin a region AND let the solver fill the rest -> C (manual + solver)")
    print(f"   * run {N} placements as parallel waves -> disjoint ROUNDS (§3), {waves} waves")
    print(hr("═"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
