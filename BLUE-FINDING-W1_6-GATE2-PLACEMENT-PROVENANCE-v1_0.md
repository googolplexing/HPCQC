# TEAM BLUE → TEAM RED — W1.6 Gate-2 FAILED: Root Cause is a Qubit-Set Provenance Mismatch, Not a Noise-Path or Harness Defect

**Document ID:** BLUE-FINDING-W1_6-GATE2-PLACEMENT-PROVENANCE-v1.0
**Date:** May 30, 2026
**From:** Team Blue
**To:** Team Red
**Re:** W1.6 production gate run (job 18943612); reconciliation ruling requested
**HEAD:** `3f53ab7` (branch `feature/device-calibrated-noise`)
**Status:** **Gate-2 ran and FAILED (52/60 kicks beyond 5σ).** Root cause diagnosed offline: the runner reference and the sweep-BYO candidate ran the device-calibrated noise on **different physical qubits** with materially different T2. **The PadDelay fix is present and working; the harness is sound.** Requesting Red's ruling on how to reconcile, plus acknowledgment of a forward-looking placement-solver design need.

---

## §1 — Summary

The 40-seed production gate (item 6 config, gated on `_w1_z_comb_verify.py`) ran clean operationally and **failed the physics gate hard**: 52 of 60 kicks beyond 5σ, worst `z_combined = 46.34`. We traced it offline. The failure is **not** a defect in the sweep-BYO device-calibrated path, **not** a regression of the PadDelay idle-decoherence fix, and **not** a harness bug. It is a **provenance mismatch in the comparison itself**: the banked runner reference self-selects its 10 physical qubits by readout fidelity, while the sweep-BYO arm runs on the placement solver's `top_1` assignment — and the two land on **different qubits**, only 5 of 10 shared, differing sharply in T2 (the channel that governs autocorrelator decay). The gate compared two physically different experiments, and correctly flagged them as inconsistent.

This note (a) shows the harness validated cleanly, (b) gives the divergence shape, (c) establishes the root cause with offline evidence, (d) names the latent assumption that broke, (e) proposes the confirmatory test, (f) requests a reconciliation ruling, and (g) raises the forward-looking placement-solver design this episode motivates.

---

## §2 — What ran, and what the harness validated (the tooling is sound)

Job 18943612, `standard`, node nid001131, elapsed 1452.8s (~24.2 min), 0 errors.

- The new `run_sweep --output-dir` override directed the **unmodified** canonical `floquet_dtc_q10_sweep.yaml` into `sweep_output/w1_gate` (YAML untouched on disk).
- Cap footer confirms the §3 sibling-map fix **live on the production run**: `cap=80; binding=num_units; 1 wave(s)`, `usable_cores_physical=128`, `per_unit_peak=1.28 GiB`, `mem_term=173`. This is carry-forward #3, captured. (The old `//2` would have shown 64 cores / 2 waves.)
- WAL consistency 80/80; 80 simulations; the candidate glob resolved to exactly one device-calibrated aggregated dat.
- The z_comb verifier computed per-kick z, returned `VERDICT: FAIL`, exited 1, and `set -e` failed the job — exactly as designed.

**One operational note (benign, recorded for the test-record):** the run emitted `WARN: memory budget fell through to node RealMemory ... safe_mem=221.0 GiB`. The launcher sets no `--mem`, so on node-exclusive `standard` the allocation-aware `SLURM_MEM_PER_NODE` path was not exercised; it fell through to RealMemory. The cap is unaffected (`min(128, 80, 128, 173)=80`) and memory was safe (MaxRSS 62.8 GiB of 221). The allocation-aware path was already proven by the D5 run (18938950); nothing for sign-off depends on this run exercising it. A future `--mem=0` on the launcher would silence the WARN.

**The harness did its job. What follows is a physics/comparison finding, not a tooling one.**

---

## §3 — The divergence shape

The candidate device-calibrated autocorrelator is systematically **larger in magnitude** than the reference at nearly every kick — it **decays more slowly** — and the gap **compounds with depth**:

| kick | cand_mean | ref_mean | z_combined |
|---|---|---|---|
| 10 | 0.5139 | 0.3887 | **46.34** (worst) |
| 40 | 0.1079 | 0.0478 | 23.29 (~2.3×) |
| 58 | 0.0560 | 0.0161 | 17.60 (~3.5×) |

There is also an **even/odd asymmetry**: the even ("return") kicks fail hard all the way out (kick 58 still z≈17.6), while the odd kicks relax back into agreement at the tail (kicks 55/57/59 at z≈0.9/1.6/1.9, passing). The candidate retains too much coherence on the return sub-sequence.

The reference's per-kick `device_cal_sem` is tiny (0.0004–0.0023, mean ≈0.0017). Since `z = |Δmean| / √(ref_sem² + cand_sem²) ≈ |Δmean| / 0.0024`, a ~1–10% systematic mean offset reads as tens of σ. The statistical sem captures shot+seed spread, not a systematic pipeline/qubit difference — so a real physical difference of a few percent saturates the 5σ bar.

---

## §4 — Root cause: the two pipelines ran on different physical qubits

**(a) The PadDelay fix is present and active — ruled out as the cause.** `90d329d` is an ancestor of HEAD; `_NATIVE_BASIS` includes `"delay"`; the runtime precondition guard (which raises if delays are unsupported) did **not** fire — the run completed. So the sweep-BYO `PadDelay`+`RelaxationNoisePass` idle channel **is** being applied. The candidate does decohere (decays 0.97 → 0.018), just more slowly than the reference. The pre-fix "idle channel entirely missing" signature is excluded.

**(b) The two pipelines self-select qubits by different criteria.** The reference is produced by `floquet_runner.py` (→ `aggregate_floquet.py` → `build_reference_csv.py`), which calls the noise builder with **no `physical_qubits`** → `_resolve_selected` falls to `_select_qubits`, the historical "best subgraph by readout fidelity" path. The sweep-BYO arm passes the **placement solver's `top_1`** assignment (the F5a placement-keyed path). Resolving both against calibration `08c3c70f`:

- **Runner reference qubits:** `QB1, QB2, QB5, QB6, QB7, QB9, QB10, QB11, QB12, QB13`
- **Sweep-BYO gate qubits:** `QB5, QB6, QB7, QB11, QB13, QB21, QB26, QB27, QB28, QB29`
- **Overlap:** 5 of 10 (`QB5, QB6, QB7, QB11, QB13`).

**(c) The non-overlapping qubits differ sharply in T2 — the dephasing time that governs decay.**

| qubit set | mean T1 | mean T2 | mean gate_err | mean readout_fid |
|---|---|---|---|---|
| runner-only (`QB1,2,9,10,12`) | 32.0 µs | **5.1 µs** | 0.00065 | 0.9697 |
| BYO-only (`QB21,26,27,28,29`) | 30.5 µs | **12.2 µs** | 0.00051 | 0.9643 |
| full runner set | 31.8 µs | **8.1 µs** | 0.00056 | 0.9733 |
| full BYO set | 31.1 µs | **11.7 µs** | 0.00049 | 0.9706 |

T1, gate error, and readout fidelity are comparable; **T2 is the differentiator** (~2.4× on the swapped-in qubits, ~1.4× overall). Higher T2 → less dephasing per idle window → slower autocorrelator decay → candidate magnitudes systematically above the reference, gap compounding over kicks. This is precisely the observed signature.

**(d) The reference was banked on the fidelity-self-selected qubits, and nobody re-checked it against the placement.** The reference CSV (commit `cb33530`, "from results/phys_*_20260524 … cal 08c3c70f, 40 inst") and its `.dat` source (`bef9c69`) come from runner runs that used `_select_qubits`, not the placement. So the comparison was never on a common qubit set.

---

## §5 — The latent assumption that broke

`FINDING-PADDELAY-SCOPE-v1_0.md` Implication 4 stated that post-fix, gate-2 should **converge** ("convergence to the reference is the measurable success criterion for the fix"). That expectation implicitly assumed **both pipelines run on the same qubits**. The F5a placement-keyed-noise feature broke that assumption: the sweep now keys device-calibrated noise to the placement solver's qubits, while the reference remained on the runner's fidelity self-selection. The gate-2 comparison is therefore **apples-to-oranges by construction** — not because either pipeline is wrong, but because they were never reconciled to a common placement after F5a landed.

---

## §6 — Residual caveat and the clean confirmatory test

We can show offline that the qubit-set mismatch is large and points the right direction, but we cannot prove offline that it accounts for **100%** of the gap — there could be a residual difference between the runner's custom Aer relaxation pass (`build_relaxation_pass`, ALAP self-scheduling) and the sweep's `PadDelay`+`RelaxationNoisePass` chain, independent of qubit choice.

**Confirmatory test (isolates the two effects):** re-run the sweep-BYO gate with `physical_qubits` pinned to the runner's self-selected set (`QB1,2,5,6,7,9,10,11,12,13`), same calibration `08c3c70f`. If it then converges within tolerance, the divergence is **purely the placement**. If a residual gap remains, that residual is the **implementation difference** between the two idle-decoherence paths — a separate, smaller finding to chase. This is one LUMI run (~25 min, same cost profile as 18943612).

---

## §7 — Disposition requested from Red

W1.6 sign-off is blocked until the comparison is reconciled. The options, for your ruling:

1. **Re-baseline the reference on the placement's qubits.** Regenerate the runner reference with `physical_qubits` pinned to the sweep-BYO `top_1` placement (`QB11,5,6,7,13,21,29,28,27,26`), so the reference and candidate share qubits. Keeps placement-keyed noise in the gate; changes the banked reference (re-anchors tolerance on those qubits).
2. **Pin the gate's placement to the runner's self-selected set** (the §6 confirmatory run, promoted to the actual gate). Keeps the existing reference authoritative; makes the gate a like-for-like reproduction on the reference's qubits; defers placement-keyed-noise validation to a separate test. *The mechanism for this is the researcher-controlled placement field proposed in BLUE-PROPOSAL-RESEARCHER-PLACEMENT-CONTROL-v1.0 (Phase 1, simulation) — a way to pin the sweep to an explicit qubit set instead of the solver's choice.*
3. **Scope placement-keyed noise out of gate-2.** Treat gate-2 as a non-placement reproduction (sweep-BYO with `physical_qubits=None`, matching the runner's self-selection), and validate F5a placement-keying under a different acceptance test.

We also request a ruling on **the cross-pipeline acceptance bar** regardless of which option above is chosen: per-kick 5σ against statistical-only sems (mean ≈0.0017) is an extremely tight bar for *any* cross-pipeline comparison — even on identical qubits, a sub-percent systematic difference between two disjoint idle-decoherence implementations could exceed it. You set z_comb-only at item 6; we ask whether a systematic-tolerance term (or a relaxed σ, or an envelope/decay-rate metric) is warranted for the cross-pipeline case, or whether the intent is strictly statistical equivalence.

Our recommendation: **run §6 first** (cheap, isolates the cause), then decide between options 1–3 with that data in hand. We have not built or changed anything toward any of these — holding for your ruling.

---

## §8 — Forward-looking: a noise-channel-aware, circuit-type-aware placement solver

This episode is concrete evidence that **the placement-selection criterion materially changes the physics result** — a 1.4–2.4× T2 swing between two "reasonable" selections moved the autocorrelator decay enough to blow a 5σ gate. That motivates a placement solver that selects qubits by **the noise channels that dominate for the circuit being run**, not by a fixed proxy.

**Why current selectors (ours and Qiskit's) are insufficient for this class of circuit.** We verified Qiskit's `VF2PostLayout` (latest docs) directly:
- It **strips idle wires** from the circuit before the subgraph match — so qubits whose role is idling are removed from the layout decision.
- Its fidelity score is built **purely from the Target's gate/readout error rates** (the injectable `vf2_avg_error_map`: avg 1q error per qubit, avg 2q error per link). There is **no T1/T2 idle-dephasing term** in the score.

For a Floquet DTC — where idle-window dephasing (T2) is the dominant decay channel, exactly what drove this gate's divergence — a VF2-style scorer optimizes gate error and is blind to the channel that actually governs the result. Our own `_select_qubits` (readout-fidelity subgraph) has the same blind spot from the other direction. Neither is "wrong"; both optimize a proxy that is mismatched to idle-dominated dynamics.

**Design direction (post-W1; leverage Qiskit's machinery, don't reinvent it).** The relevant catalog (qiskit.transpiler.passes):
- *Scheduling* — `ALAPScheduleAnalysis` / `ASAPScheduleAnalysis` + `PadDelay` give us the per-qubit **idle-duration profile** of a transpiled circuit. That is the missing ingredient: weight each qubit's T1/T2 by how long it actually idles in *this* circuit.
- *Circuit Analysis* — `Depth`, `CountOps`, `DAGLongestPath` classify the circuit (gate-heavy vs idle-heavy) so the scorer can weight channels by what dominates.
- *Post Layout* — `VF2PostLayout` accepts a custom `vf2_avg_error_map` via the property set. That is the clean seam: an upstream analysis pass builds a **circuit-type-aware error map** that folds idle dephasing (T1/T2 × idle duration) into the per-qubit term alongside gate/readout error, then VF2 does the subgraph search against *that*. We get VF2's solver for free and supply the physics-correct objective.

**Dual purpose, which is the point.** The same noise-aware scorer produces a **ranked list of top placements** usable for both (i) which qubits to *simulate* (sim placement selection, sorting candidates by predicted fidelity for the circuit) and (ii) which physical qubits to *run on Q50* for real-hardware execution. That makes the three-way comparison the program wants — noiseless sim vs noisy sim vs QPU — well-posed on a common, defensible placement, rather than each path self-selecting differently (which is exactly the failure mode this note documents).

**Interaction with §7.** This is post-W1 and does not block current closure, but the gate-2 reconciliation choice should be made with it in mind: option 1 (placement-keyed reference) is the path that keeps placement in the loop and lines up with this future solver; options 2/3 defer placement validation. We flag the dependency so the W1.6 decision doesn't foreclose the forward design.

**A prerequisite the solver work shares: researcher-controlled placement.** Whether the future solver picks qubits well or not, a researcher must be able to *override* it — to replicate a prior experiment on its original qubits (a device-calibrated noise model needs both the calibration snapshot, already selectable, and the qubits, currently not), to try alternative selection heuristics, and to compare solver-choice vs experiment-choice vs hand-choice on the same circuit. That capability is specified separately in BLUE-PROPOSAL-RESEARCHER-PLACEMENT-CONTROL-v1.0 and tracked as DEBT `PLACEMENT-1`; its Phase 1 (simulation) is also the mechanism for §7 option 2 above.

---

## §9 — Asks

1. **Ruling on §7** — reconciliation option (1/2/3) and the cross-pipeline acceptance-bar question. Our recommendation: authorize the §6 confirmatory run first.
2. **Acknowledgment of §8** as a tracked post-W1 forward item (noise-channel-aware, circuit-type-aware placement solver; VF2PostLayout's gate-error-only scoring and idle-wire stripping recorded as the prior-art pitfall to design around).

No code or config changed toward any of the above. The harness, the cap fix, and the `--output-dir` override are validated and stand; only the comparison's placement basis is in question.

*— Team Blue. Gate-2 executed; failure root-caused to a qubit-set provenance mismatch (T2 1.4–2.4×), not a noise-path or harness defect; reconciliation ruling requested; forward placement-solver need flagged with the VF2PostLayout limitation verified.*
