#!/usr/bin/env python3
"""Verify BYO chain placements against a device calibration BEFORE a survey run.

A placement is REJECTED if any of its qubits is absent from cal["qubits"] or has
a null critical field (t1_us/t2_us/t2_echo_us/readout_fidelity), or if any
consecutive pair is not a device edge in cal["two_qubit_gates"] (either
orientation) or has a null cz_fidelity/cz_error. These are exactly the lookups
device_noise.build_* performs, so a rejected placement is one that would KeyError
or TypeError the device-calibrated noise build mid-shard.

Usage:  python3 verify_placements_vs_cal.py <calibration.json> <config.yaml>
Exit 0 if all placements valid; 1 if any are invalid (prints the offenders).
"""
import json
import re
import sys

CRIT = ["t1_us", "t2_us", "t2_echo_us", "readout_fidelity"]


def main(cal_path, yaml_path):
    cal = json.load(open(cal_path))
    qubits = cal["qubits"]
    tq = cal["two_qubit_gates"]

    def qubit_problem(name):
        d = qubits.get(name)
        if d is None:
            return "absent from cal['qubits']"
        miss = [f for f in CRIT if d.get(f) is None]
        return f"null/missing {miss}" if miss else None

    def edge_problem(a, b):
        for key in (f"{a}-{b}", f"{b}-{a}"):
            g = tq.get(key)
            if g is not None:
                if g.get("cz_fidelity") is None or g.get("cz_error") is None:
                    return f"{key}: null cz"
                return None
        return f"{a}-{b}: not a device edge"

    # Parse placements: every "- [QB.., QB.., ...]" whose entries are all QB ids.
    text = open(yaml_path).read()
    placements = []
    for m in re.finditer(r"-\s*\[([^\]]+)\]", text):
        qs = [x.strip() for x in m.group(1).split(",")]
        if qs and all(re.fullmatch(r"QB\d+", x) for x in qs):
            placements.append(qs)

    invalid = []
    used = set()
    for i, chain in enumerate(placements):
        used.update(chain)
        probs = []
        for q in chain:
            p = qubit_problem(q)
            if p:
                probs.append(f"qubit {q}: {p}")
        for a, b in zip(chain, chain[1:]):
            p = edge_problem(a, b)
            if p:
                probs.append(f"edge {p}")
        if probs:
            invalid.append((i, chain, probs))

    print(f"calibration : {cal.get('calibration_set_id')}  ({cal.get('timestamp')})")
    print(f"placements parsed : {len(placements)}")
    print(f"distinct qubits used : {len(used)}  "
          f"(QB32 used: {'YES' if 'QB32' in used else 'no'})")
    # also report chain length consistency
    lens = sorted({len(c) for c in placements})
    print(f"chain lengths : {lens}")
    print(f"INVALID placements : {len(invalid)}")
    for i, chain, probs in invalid:
        print(f"  [{i}] {'-'.join(chain)}")
        for p in probs:
            print(f"       {p}")
    return 1 if invalid else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
