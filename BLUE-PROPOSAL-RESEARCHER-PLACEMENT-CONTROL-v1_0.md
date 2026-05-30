# TEAM BLUE → TEAM RED — PROPOSAL: Researcher-Controlled Placement Selection (Manual Multi-Placement via YAML)

**Document ID:** BLUE-PROPOSAL-RESEARCHER-PLACEMENT-CONTROL-v1.0
**Date:** May 30, 2026
**From:** Team Blue
**To:** Team Red
**Re:** A YAML seam to supply explicit physical-qubit placements (one or many), bypassing the solver — for simulation now (Phase 1) and Q50 QPU execution later (Phase 2)
**HEAD:** `3f53ab7` (branch `feature/device-calibrated-noise`)
**Related:** BLUE-FINDING-W1_6-GATE2-PLACEMENT-PROVENANCE-v1.0 (this is the mechanism for that note's §7 option 2); DEBT entry `PLACEMENT-1`
**Status:** PROPOSAL — requesting Red's ruling on adopting the capability and the Phase-1/Phase-2 staging. No code changed.

---

## §1 — Summary

Today a BYO sweep's physical qubits are an **output** of the placement solver (`self._solver.find_all_placements(..., strategy="max_fidelity")`), never an input. We propose a YAML field that lets a researcher supply **explicit placements as lists of qubit strings** — one or several — which bypass the solver and feed those exact qubit sets through the existing execution paths.

The capability is **imperative for reproducibility** (§2), not merely a convenience. It is also the concrete mechanism for the gate-2 reconciliation (§7 option 2 of the findings doc).

The work splits cleanly, and we verified the split against the tree rather than assuming it (§6):

- **Phase 1 (simulation) is a front-end seam, mostly plumbing.** Task multiplication by placement and cap-based parallel execution already work; BYO just always calls the solver. Intercepting that call with a YAML-supplied placement list is small and self-contained.
- **Phase 2 (Q50 QPU) is an integration, not plumbing.** The packing algorithm (`MixedPacker`) and the IQM/VTT backend (`iqm_qpu`, `VTT_BATCH_LIMIT=100`) exist and are individually tested, **but neither is wired into the BYO path — BYO is simulation-only today.** Delivering the QPU packing/batching semantics is real integration and should be staged separately.

---

## §2 — Motivation: this is a reproducibility requirement

A device-calibrated noise model is fully determined by **two independent inputs**: the **calibration snapshot** (the T1/T2/gate/readout parameters of an era) and the **physical qubits** those parameters are keyed to. The framework already lets a researcher control the first — the YAML `calibrations:` field accepts any calibration JSON, including an older one — but **not** the second: qubit choice is a solver output.

That asymmetry breaks replication exactly where device-calibrated work needs it. To faithfully reproduce a prior experiment you must reproduce **both** halves; the solver, by design (`max_fidelity`), picks current-optimal qubits, which in general are **not** the qubits a past run used — different optimum, different calibration era, or the original researcher's own non-optimal choice. So today you can match an experiment's calibration but not its qubits, and therefore cannot reproduce its placement-keyed noise.

**Gate-2 is the live proof.** The solver's `top_1` (`QB11…QB26`) was not the runner reference's fidelity-self-selected set (`QB1,2,5,6,7,9,10,11,12,13`); the device-calibrated arms diverged at tens of σ, driven by a ~2.4× T2 difference on the swapped qubits (see the findings doc). Replication demands the qubit set be an input the researcher supplies.

Beyond replication, manual selection serves the everyday cases: trying alternative selection heuristics, comparing **solver-choice vs experiment-choice vs hand-choice** on the same circuit, and — once the noise-aware solver lands (findings doc §8) — letting a researcher accept, reject, or override its ranked placements.

---

## §3 — Proposed YAML schema

A new optional field on the BYO experiment, accepting a **list of placements**, each a list of qubit-name strings:

```yaml
sweep:
  experiments:
    - type: byo_circuit
      label: floquet_dtc_q10_replication
      circuit_script: examples/byo/floquet_dtc.py
      fixed: {num_qubits: 10, epsilon: 0.03}
      grid: {num_kicks: {range: [0, 60]}}
      seed_list: [0, 1, 2, 3]
      shots: 1000
      noise_configs: [noiseless, device_calibrated]   # all, or a channel subset
      physical_qubits:                                  # NEW — bypasses the solver
        - [QB1, QB2, QB5, QB6, QB7, QB9, QB10, QB11, QB12, QB13]   # placement A
        - [QB11, QB5, QB6, QB7, QB13, QB21, QB29, QB28, QB27, QB26] # placement B
  calibrations:
    - examples/q50_calibration_20260524_08c3c70f.json
```

Semantics:
- **Field present** → the solver is bypassed; each inner list is used **verbatim, in the given logical order** (logical qubit `i` → `physical_qubits[k][i]`), exactly as the F5a `_resolve_selected` placement path already does for a single placement.
- **Field absent** → unchanged current behavior (solver `find_all_placements`, `top_1` device-cal guardrail). Fully backward compatible.
- **Composition** is orthogonal: `physical_qubits` (which qubits) × `calibrations` (which era) × `noise_configs` (noiseless / all channels / subset) × `seed_list` × `grid.num_kicks` jointly determine the circuit/task set. Each placement participates exactly as a solver placement would.

A single-element list (one placement) is the common replication case; multiple elements enable the multi-placement study below.

---

## §4 — Execution semantics

### §4.1 — Simulation (Phase 1)
Placements multiply units: `units = placements × seeds × envs`. One manual placement at 40 seeds × 2 envs = 80 units (the gate's shape); two placements, all else equal, = 160 units. Units run in parallel under the W1 allocation-aware cap (the same forkserver pool just exercised at `cap=80, 1 wave`). Doubling placements doubles units; the cap schedules them across the node — no new execution machinery.

### §4.2 — Q50 QPU (Phase 2)
The semantics the researcher described map onto the existing `MixedPacker` model:
- **Non-overlapping placements share one device submission.** `MixedPacker` packs `(circuit, placement)` combos that share **neither physical qubits nor coupling edges** into a single round — i.e. disjoint placements run together on the device. (Note: the existing packer checks **edge** overlap as well as qubit overlap, so it is correctly stricter than "unique qubits": adjacent disjoint placements that would crosstalk are not packed together.)
- **Overlapping placements run as separate circuits.** Any qubit/edge conflict forces the placements into different circuits.
- **Batches cap at the device policy (default 100).** `iqm_qpu.VTT_BATCH_LIMIT` is auto-detected from the VTT/IQM `max_number_circuits_per_batch` policy, fallback 100, overridable via `qpu.batch_limit`. Rounds are submitted within that limit.

**Honest scope note:** these are tested *components*, not a connected BYO→QPU pipeline. Phase 2 is the integration that routes BYO circuits → `MixedPacker` (rounds) → `iqm_qpu._submit_batch` (≤ batch limit), with the overlap→separate-circuit fallback. See §6.

---

## §5 — Validation (fail-loud)

`_resolve_selected` already validates a **single** placement: count must equal `num_qubits`, every qubit name must exist in the calibration, and any supplied edge must be a real calibrated two-qubit gate among those qubits (RED-REVIEW-SPEC-002-7.5 §3.1). The proposal **extends this list-wise**, applying the same checks per placement, plus:
- Each inner list length == `fixed.num_qubits`.
- Every name resolvable in the active calibration (drift between a researcher's chosen qubits and an older calibration is caught here, not silently mis-keyed).
- For QPU (Phase 2): cross-placement qubit/edge overlap is **detected** (it routes packing vs separate-circuit), not rejected — overlap is legal input with defined behavior.

Fail-loud is essential precisely for the replication case: a typo'd qubit name or a name absent from a chosen older calibration must raise, never silently fall back to the solver.

---

## §6 — Reuse map (verified against HEAD `3f53ab7`)

| Component | State | Evidence |
|---|---|---|
| Placement → task/unit multiplication | **EXISTS & WIRED** (sim) | BYO unit model; gate ran 40×2×1 = 80 units |
| Cap-based parallel sim execution | **EXISTS & WIRED** | W1 forkserver cap; gate `cap=80, 1 wave` |
| Calibration selection (incl. older) | **EXISTS & WIRED** | YAML `calibrations:` field |
| Noise selection (all / channel subset) | **EXISTS & WIRED** | `noise_configs: "all" | [list]` + NoiseConfig envs |
| Single-placement fail-loud validation | **EXISTS & WIRED** | `_resolve_selected` (F5a path) |
| `MixedPacker` (qubit+edge overlap → rounds) | **EXISTS, NOT WIRED** | `mixed_packing.py`; only exported + `tests/e6b_mixed_packing_validation.py` — no live caller |
| IQM/VTT backend + 100-circuit batch | **EXISTS, NOT WIRED to BYO** | `iqm_qpu.py` `VTT_BATCH_LIMIT=100`, `_submit_batch`; `QPUConfig` parses `qpu:` |
| BYO → QPU execution path | **DOES NOT EXIST** | `_execute_byo_group` is forkserver/Aer only; no device/qpu branch |
| **YAML `physical_qubits` multi-placement field** | **NEW (Phase 1)** | — |
| **Solver-bypass injection at `find_all_placements`** | **NEW (Phase 1)** | — |
| **List-wise validation extension** | **NEW (Phase 1)** | — |
| **BYO → MixedPacker → iqm_qpu integration** | **NEW (Phase 2)** | — |

The takeaway: Phase 1 is three small additions on top of wired, tested machinery; Phase 2 is a genuine integration of two tested-but-unconnected subsystems.

---

## §7 — Proposed staging

**Phase 1 — manual multi-placement for simulation (small, do first).**
YAML field + solver-bypass at the `find_all_placements` call site in `_execute_byo_group` + list-wise validation. Unblocks: (a) the gate-2 reconciliation option 2 / the confirmatory run (pin the sweep to the runner's qubit set), and (b) calibration-matched **replication in simulation** — the imperative in §2 — immediately.

**Phase 2 — Q50 QPU packing/batching (larger, later).**
Integrate manual (and solver) placements through `MixedPacker` and the `iqm_qpu` backend: pack disjoint placements into shared rounds, route overlapping placements to separate circuits, submit within the VTT batch limit. Sized as integration, with its own validation and its own canary against a known device submission.

Phase 1 alone satisfies the replication requirement for the simulation workflow, which is where W1 and the gate-2 reconciliation live. Phase 2 is the on-hardware extension and can follow the noise-aware-solver work (findings doc §8) it naturally pairs with.

---

## §8 — Relationship to the gate-2 reconciliation

This proposal is the mechanism for **option 2** in BLUE-FINDING-W1_6-GATE2-PLACEMENT-PROVENANCE §7 (pin the gate's placement to the runner's self-selected set so the comparison is like-for-like) and for the §6 confirmatory run there. Phase 1 is sufficient for both. If Red instead chooses option 1 (re-baseline the reference on the placement's qubits), this field still applies — it's how the runner reference would be pinned to a chosen placement. Either reconciliation path benefits from the capability; it does not presuppose a particular ruling.

---

## §9 — Asks

1. **Ruling on adoption** of researcher-controlled placement selection as specified (§3 schema, §4 semantics, §5 validation).
2. **Ruling on the Phase-1/Phase-2 staging** (§7) — in particular, authorization to build Phase 1 (simulation) as the near-term, gate-2-unblocking increment, with Phase 2 (QPU) deferred and separately reviewed.
3. **Acknowledgment** that calibration selection already exists, so the capability closes the second half of the replication input pair (qubits) rather than adding a standalone knob.

No code or config changed. On a ruling, Phase 1 is a small, testable increment (front-end seam + list validation + unit tests, all offline-verifiable); Phase 2 would come back as its own design.

*— Team Blue. Manual multi-placement proposed as a reproducibility requirement; simulation path is a verified front-end seam, QPU path is a verified-but-unwired integration staged separately; reuse map grounded against HEAD, not assumed.*
