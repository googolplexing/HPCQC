#!/usr/bin/env python3
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Generate diverse, device-covering N-qubit chains for a QPU-surface DTC survey.

Builds self-avoiding paths on the calibration coupling graph (dead/uncalibrated
qubits excluded), then greedily selects a set that (a) covers every usable qubit
at least k_min times and (b) is geometrically diverse (deduped by qubit set,
balanced across the device). Prints a coverage histogram and writes the chains as
a YAML physical_qubits block ready to paste into the sweep config.

  python generate_qpu_survey_chains.py examples/q50_calibration_20260524_08c3c70f.json \\
      --length 10 --kmin 15 --out survey_chains.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict


def build_graph(cal_path):
    d = json.load(open(cal_path))
    cal_qubits = set(d["qubits"])  # only qubits WITH calibration data are usable
    adj = defaultdict(set)
    for a, b in d.get("qubit_connectivity", []):
        if a in cal_qubits and b in cal_qubits:
            adj[a].add(b); adj[b].add(a)
    usable = sorted(cal_qubits & set(adj), key=lambda s: int(s[2:]))
    return adj, usable


def random_saw(adj, length, start, rng):
    """One random self-avoiding walk of `length` nodes, or None if it dead-ends."""
    path = [start]
    seen = {start}
    while len(path) < length:
        nbrs = [n for n in adj[path[-1]] if n not in seen]
        if not nbrs:
            return None
        nxt = rng.choice(nbrs)
        path.append(nxt); seen.add(nxt)
    return tuple(path)


def generate(adj, usable, length, kmin, max_chains, rng, oversample=60):
    """Greedy coverage-balancing selection from a large SAW candidate pool."""
    # Candidate pool: many SAWs from every start node, deduped by frozenset.
    pool = {}
    for _ in range(oversample):
        for s in usable:
            w = random_saw(adj, length, s, rng)
            if w:
                pool[frozenset(w)] = w
    candidates = list(pool.values())
    rng.shuffle(candidates)

    chosen = []
    cover = Counter()
    # Phase 1: greedily satisfy kmin coverage (pick the chain helping the most
    # under-covered qubits each step).
    def deficit_gain(chain):
        return sum(max(0, kmin - cover[q]) > 0 for q in chain)
    remaining = candidates[:]
    while remaining and min((cover[q] for q in usable), default=kmin) < kmin \
            and len(chosen) < max_chains:
        remaining.sort(key=deficit_gain, reverse=True)
        best = remaining.pop(0)
        if deficit_gain(best) == 0:
            break
        chosen.append(best)
        for q in best:
            cover[q] += 1
    # Phase 2: top up with diverse chains until max_chains (diversity = pick the
    # chain maximizing coverage of the currently-least-covered qubits).
    while remaining and len(chosen) < max_chains:
        remaining.sort(key=lambda c: sum(1.0 / (1 + cover[q]) for q in c),
                       reverse=True)
        best = remaining.pop(0)
        chosen.append(best)
        for q in best:
            cover[q] += 1
    return chosen, cover


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("calibration")
    ap.add_argument("--length", type=int, default=10)
    ap.add_argument("--kmin", type=int, default=15, help="min chains covering each qubit")
    ap.add_argument("--max-chains", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="survey_chains.yaml")
    args = ap.parse_args(argv)

    rng = random.Random(args.seed)
    adj, usable = build_graph(args.calibration)
    chains, cover = generate(adj, usable, args.length, args.kmin,
                             args.max_chains, rng)

    cov_vals = [cover[q] for q in usable]
    print(f"usable qubits: {len(usable)}  (excluded: no-calibration qubits)")
    print(f"chains chosen: {len(chains)}  (length {args.length})")
    print(f"coverage per qubit: min={min(cov_vals)} mean={sum(cov_vals)/len(cov_vals):.1f} "
          f"max={max(cov_vals)}")
    under = [q for q in usable if cover[q] < args.kmin]
    print(f"qubits below kmin={args.kmin}: {len(under)} {under[:8]}")
    # distinct geometries: unique qubit sets / unique edge multisets
    print(f"distinct qubit-sets: {len(set(frozenset(c) for c in chains))}")

    with open(args.out, "w") as f:
        f.write("      physical_qubits:\n")
        for c in chains:
            f.write("        - [" + ", ".join(c) + "]\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
