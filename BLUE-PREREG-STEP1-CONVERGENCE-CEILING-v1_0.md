# TEAM BLUE → TEAM RED — PRE-REGISTRATION: Step-1 Convergence Ceiling (before the run)

**Document ID:** BLUE-PREREG-STEP1-CONVERGENCE-CEILING-v1.0
**Date:** May 30, 2026
**From:** Team Blue
**To:** Team Red
**Re:** The pre-committed convergence ceiling for the §4 Step-1 confirmatory run, registered **before** the run per RED-CLARIFICATION-STEP1-SIGMA-SYS §2–§3
**HEAD:** `d3f1b3b` (Phase-1 seam, confirmed: 18/18 unit, job 18946498) + this commit, which adds the `step1-residual` verifier mode and its unit test
**Related:** RED-RESP-GATE2-…-RULING §4/§5; RED-CLARIFICATION-STEP1-SIGMA-SYS; BLUE-FINDING-W1_6-GATE2-PLACEMENT-PROVENANCE
**Status:** PRE-REGISTRATION — requesting Red's approval of the `floor` and ceiling **numbers** before Step 1 is run. No run has been submitted.

---

## §1 — Why this exists

RED-CLARIFICATION §2/§5: Step 1's convergence call must be made against a ceiling that is **fixed before the data exists**, or "converges" is an eyeball judgment. The ceiling is the **defect line**; it is a *different quantity* from σ_sys (the measured residual): **ceiling ≥ σ_sys**, the ceiling is pre-registered, σ_sys is whatever the run produces. This document pins the ceiling now so the convergence verdict cannot be tuned to the result.

## §2 — What Step 1 is

Re-run the sweep-BYO gate with `physical_qubits` pinned to the runner's self-selected set `QB1,2,5,6,7,9,10,11,12,13`, calibration `08c3c70f`, 40 seeds — i.e. **both arms on identical qubits**. The reference is the existing runner CSV (`examples/reference/floquet_dtc_q10_autocorr.csv`, `cb33530`), already banked on those same qubits, so **no reference regeneration** is needed (that is Step 2 / Option 1). One ~25-min `standard` job. The residual it measures is the cross-implementation difference between the runner's ALAP self-scheduling and the sweep's `PadDelay`+`RelaxationNoisePass` idle model — which **becomes σ_sys** for the Option-1 gate.

## §3 — The metric (not sem-normalized)

Per §5 of the ruling, the statistical z is the wrong bar here (a sub-percent systematic offset reads as tens of σ against the ~0.0017 sems regardless of correctness). Step 1 uses a **relative-deviation envelope**:

- Per-kick `rel_dev(k) = |cand(k) − ref(k)| / max(|ref(k)|, floor)`.
- **`floor = 0.02`.** The autocorrelator decays to ~0.018 at the tail and the odd ("non-return") kicks sit near zero; below ~0.02 the signal is at the noise level and a relative metric is meaningless. The floor excludes those kicks from the rel-dev summary rather than letting them dominate it. Above-floor kicks are the even/return-echo envelope.
- **Decay-rate diagnostic:** an OLS fit of `log|A(k)|` over the even kicks above floor, for both arms; the relative difference of the extracted decay rates is reported. The physics question is "do they decay the same way," so this is the corroborating physics check.
- **Depth-trend diagnostic:** the OLS slope of `rel_dev` vs kick. The gate-2 failure *compounded* with depth (≈2.3× at kick 40, ≈3.5× at kick 58); a benign implementation difference on identical qubits should be roughly depth-stable, so a positive trend is a warning sign even under the magnitude bound.

## §4 — The pre-committed ceiling (the number requiring approval)

**Convergence iff `max_k rel_dev(k) ≤ 2.0%` over above-floor kicks**, with the depth-trend and decay-rate diagnostics consistent with a flat, non-compounding offset (Red reviews the diagnostics; the hard binary is the 2% bound, which already catches compounding because a compounding residual exceeds it at the tail).

Justification of 2%: on **identical qubits with identical calibration**, both arms apply the same T1/T2 relaxation channels; the only difference is idle-scheduling granularity (ALAP self-scheduling vs `PadDelay`-inserted delays + `RelaxationNoisePass`). The physical expectation is **sub-percent**; 2% is deliberate headroom, not a target. We expect the measured σ_sys to come in well under it.

**This is the defect line, not σ_sys.** If the measured residual **exceeds** 2%, the response is to investigate and fix the idle-decoherence implementation difference — **not** to raise the ceiling or inflate σ_sys (RED-CLARIFICATION §5). A residual that large means the two idle models genuinely disagree, which must be understood before any placement-keyed reference (Option 1) is built on top of the sweep's model.

## §5 — Forward to the Option-1 gate (σ_sys transfer)

Per RED-CLARIFICATION §4: the ceiling and metric are **relative** (a fraction of the signal), which transfers between qubit sets more defensibly than an absolute constant — Step 1 measures on the runner qubits, the Option-1 gate runs on the placement qubits `QB11,5,6,7,13,21,29,28,27,26`. The Option-1 gate is itself an identical-qubit comparison (reference re-baselined on the placement qubits), so it yields a **second** measurement of the residual; if that placement-qubit residual is consistent with the Step-1 (runner-qubit) σ_sys, the systematic term is qubit-stable and the gate is sound. If it jumps, that is a qubit-dependent implementation difference — a finding to understand, not absorb. The augmented-z gate (`z = |Δ| / √(ref_sem² + cand_sem² + σ_sys²)`) parameterized by the measured σ_sys is a **later** addition to the verifier; it is not part of Step 1.

## §6 — Mechanism (already built, offline-verified)

`tests/_w1_z_comb_verify.py --mode step1-residual` implements the above (gate mode unchanged; `--floor`, `--max-rel-dev` parameterized; exit 0 = converged, 1 = over ceiling, 3 = structural). Cross-checked by `tests/unit/test_w1_step1_residual.py` (relative-deviation + floor, decay-rate fit, residual summary, and the converged/over-ceiling verdict). Pure stdlib; runs on a login node or in the container.

## §7 — Ask

1. **Approve `floor = 0.02`** (the noise-floor cutoff for the relative metric).
2. **Approve the ceiling `max rel_dev ≤ 2.0%`** as the pre-committed convergence defect line for Step 1.
3. On approval, we pre-register these numbers in the W1 test-record and run Step 1; we report the measured σ_sys (max / mean rel-dev, depth-trend, decay-rate difference) and, if converged, carry σ_sys forward to the Option-1 gate. If it exceeds the ceiling, we stop and open the idle-implementation finding.

*— Team Blue. Ceiling pre-registered before the data; ceiling ≠ σ_sys; σ_sys is the measured output. No run submitted pending Red's approval of the two numbers.*
