# FINDING-PADDELAY-SCOPE

**Version:** 1.0
**Date:** 2026-05-27
**Branch:** `feature/device-calibrated-noise`
**HEAD at finding:** `f0ce463`
**Pre-fix tag:** `pre-paddelay-fix` (commit `fe22fe5`)
**Fix commit:** `90d329d`
**Related:** `FINDING-PADDELAY-IDLE-NOT-INSERTED-v1_0.md`,
`RED-VERIFY-PADDELAY-IDLE-NOT-INSERTED-v1_0.md` (the verification that
approved the fix and prescribed §6 re-baselining)

---

## Summary

The PadDelay fix at 90d329d corrects idle-time decoherence on **one** of two
architecturally separate idle-decoherence pipelines in this repository, not
both. The empirical re-baseline run (job 18874828, post-fix device-calibrated
reproduction with calibration `08c3c70f`) produced an aggregated autocorrelator
**byte-identical** to the pre-fix banked reference at
`examples/reference/floquet_dtc_q10_device-cal_agg.dat` — because the reference
is produced by `floquet_runner.py`, whose idle-decoherence path never went
through the buggy `prepare.py` code.

Conclusion: the §6 scope statement in
`RED-VERIFY-PADDELAY-IDLE-NOT-INSERTED-v1_0.md` ("all device-calibrated
reference artifacts on this branch invalidated and must be re-baselined post-fix
before D3.5 gate-2 can re-anchor tolerance") is narrower than originally stated.
Artifacts produced via the **sweep BYO path** pre-fix would have been invalid;
artifacts produced via the **`floquet_runner.py` reference path** were not
affected by the bug. The committed reference at
`examples/reference/floquet_dtc_q10_device-cal_agg.dat` (and its noiseless
companion) remains intact and authoritative for D3.5 gate-2 tolerance anchoring.

---

## The two pipelines

| Aspect | Reference path (`floquet_runner.py`) | Sweep BYO path (`sweep_engine.py`) |
|---|---|---|
| Scheduler / Delay insertion | `_prepare_device_circuits()` in `floquet_runner.py:156` | Qiskit `PadDelay` in `prepare.py` (via `prepare_simulation`) |
| Idle decoherence channel | `device_noise.build_relaxation_pass()` (`device_noise.py:339`), registered as `noise_model._custom_noise_passes` (Aer custom pass, runs at assemble time on any circuit containing Delays) | `RelaxationNoisePass(op_types=[Delay])` (Qiskit transpiler pass) |
| Dependency on `_NATIVE_BASIS` | None (own scheduling, custom pass) | Yes — `PadDelay.__delay_supported(q)` consults the `Target` built from `_NATIVE_BASIS`; missing `"delay"` returned False on every qubit |
| Bug status | **Never had the bug** | Had the bug; fixed at 90d329d |

The runner pipeline is deliberately a self-contained "reference" execution path
that mirrors the researcher's banked configuration; it does its own scheduling
and attaches its idle-decoherence pass as a Aer-side NoiseModel hook. The sweep
BYO path was added later as the reproducible production-scale execution surface
and rebuilt on Qiskit's standard `PadDelay`+`RelaxationNoisePass` toolchain. The
two ended up architecturally disjoint, which is why a bug confined to the
latter's `Target`-construction in `prepare.py` left the former unaffected.

---

## Empirical evidence

**1. Code-path verification (commit f0ce463, LUMI working tree):**

```
$ grep -n "build_relaxation_pass\|_prepare_device_circuits\|prepare_simulation\
|_NATIVE_BASIS\|PadDelay" floquet_runner.py
46:        build_relaxation_pass,
59:            build_relaxation_pass,
64:        build_relaxation_pass = None
156:def _prepare_device_circuits(circuits, num_qubits, calibration_path, t2_mode,
254:        relax_pass, _, _ = build_relaxation_pass(
339:            run_circuits, simulator, relax_pass, dinfo = _prepare_device_circuits(

$ grep -n "def build_relaxation_pass" src/lumi_hpc_qc/backends/device_noise.py
339:def build_relaxation_pass(
```

Zero matches in `floquet_runner.py` on `prepare_simulation`, `_NATIVE_BASIS`,
or `PadDelay`. The runner does not consume the buggy code path.

**2. Byte-identical aggregate (post-fix run vs pre-fix banked reference):**

Job 18874828 — device-calibrated reproduction at commit f0ce463 (post-fix HEAD),
40 instances × 60 kicks × 1000 shots, calibration
`examples/q50_calibration_20260524_08c3c70f.json`, master_seed=0,
initial_state=3, pool wall 667.89s:

```
$ sha256sum results/floquet_device_calibrated_20260527_212801_job18874828/aggregated_autocorr.dat \
            examples/reference/floquet_dtc_q10_device-cal_agg.dat
aa7084f2df8a7cb16ae981ed03aed7e726f6f5f893fd52006a19dd1bda5841a3  results/floquet_device_calibrated_20260527_212801_job18874828/aggregated_autocorr.dat
aa7084f2df8a7cb16ae981ed03aed7e726f6f5f893fd52006a19dd1bda5841a3  examples/reference/floquet_dtc_q10_device-cal_agg.dat
```

Both files identical at the bit level. The `diff` is empty.

**3. Noiseless reference also stable:**

```
$ sha256sum examples/reference/floquet_dtc_q10_noiseless_agg.dat
d280052001129511b0a379ddd4a37d1ad709c109b156331a8905297540fc39b8  examples/reference/floquet_dtc_q10_noiseless_agg.dat
```

Matches the SHA256 from the post-fix F4 noiseless reproduction reported as
satisfying `RED-VERIFY-PADDELAY-IDLE-NOT-INSERTED-v1_0.md` §5 #3 (F4 noiseless
byte-identicality smoke). The noiseless path has no decoherence channels at all,
so it would not exercise the PadDelay bug either way; this datapoint is a
secondary consistency check, not a new claim.

**4. Provenance of the committed reference:**

```
$ git log --oneline -- examples/reference/floquet_dtc_q10_device-cal_agg.dat
bef9c69 chore(provenance): F4 banked reference from results/phys_*_20260524 (master_seed=0, initial_state=3, cal 08c3c70f, 40 inst) + DEBT.md (RED-VERIFY §5)
```

Single commit, banked from a runner-produced results directory dated 2026-05-24
— well before the PadDelay fix at 90d329d. The pre-fix runner produces the
same bits as the post-fix runner because the bug never affected either.

**5. FFT + autocorrelator plots:** the corrected-regime aggregate exhibits the
expected DTC signature — sharp FFT peak at f = 0.5 (1/T) with amplitude ≈ 0.14
and symmetric Lorentzian falloff; period-2 autocorrelator alternation preserved
through 60 kicks with smooth exponential-like envelope from ~0.97 to ~0.02.
Saved alongside `aggregated_autocorr.dat` in the run's OUTDIR.

---

## Scope correction to `RED-VERIFY-PADDELAY-IDLE-NOT-INSERTED-v1_0.md` §6

§6 originally read (paraphrased):

> "all device-calibrated reference artifacts on this branch invalidated and
> must be re-baselined post-fix before D3.5 gate-2 can re-anchor tolerance"

Empirically narrowed scope, given the architectural disjointness above:

> Artifacts that were produced through the sweep BYO path pre-fix would have
> been invalid (under-decohered, missing the idle-time channel). Artifacts
> produced through the `floquet_runner.py` reference path were not affected by
> the bug. No sweep-BYO-produced artifacts are committed at the time of this
> finding; the committed reference at
> `examples/reference/floquet_dtc_q10_device-cal_agg.dat` was produced by the
> reference path, is byte-identical to the corresponding post-fix re-baseline
> run, and remains the authoritative target for D3.5 gate-2 tolerance anchoring.

The hazard class recorded in `DEBT.md` D8 ("third-party library precondition
becomes in-repo invariant") is unchanged by this finding — the precondition
assertion and regression tests landed at 90d329d still apply, and the
under-decohered-output failure mode they prevent would still bite any new
artifact path that consumes `prepare.py` if `_NATIVE_BASIS` regressed.

---

## Implications

1. **No re-baseline commit required.** The replacement-and-CHANGELOG-shift
   workflow described between the verification and this finding becomes moot.
   `examples/reference/floquet_dtc_q10_device-cal_agg.dat` is not touched.
   `examples/reference/floquet_dtc_q10_autocorr.csv` is not regenerated.

2. **D3.5 gate-2 tolerance basis is unchanged.** The per-kick sem of the
   existing reference CSV remains the tolerance anchor.

3. **The PadDelay fix at 90d329d is still correct and still necessary** — but
   for the sweep BYO path only. Pre-fix, the BYO sweep silently produced
   under-decohered device-calibrated output; post-fix, it produces output that
   includes the idle-time decoherence channel.

4. **D3.5 gate-2 is now a meaningful test of the fix.** Gate-2 compares the
   sweep BYO path's aggregated autocorrelator against the
   `floquet_runner.py`-banked reference. Pre-fix, the BYO sweep would have
   missed the idle-time channel entirely (a visible departure from the
   reference at later kicks where idle gaps accumulate). Post-fix, the BYO
   sweep applies idle-time decoherence via the now-functioning
   `PadDelay`+`RelaxationNoisePass` chain; convergence to the reference is the
   measurable success criterion for the fix.

5. **Pipeline diversity note for future work.** The two pipelines applying
   nominally the same physics through different code paths is a useful
   internal consistency check, but is also a maintenance hazard: a bug that
   touches one pipeline's idle-decoherence may not surface in the other's
   output, and a divergence between them is ambiguous (could be a real
   physics modeling difference or a regression in one path). Consolidating to
   a single shared scheduler + idle-decoherence implementation, used by both
   `floquet_runner.py` and the sweep BYO path, would remove this ambiguity.
   Tracked as a follow-on consideration; out of scope for this finding.

---

## Forward path

1. **Update the `CHANGELOG.md` entry for the PadDelay fix** to reflect the
   correct scope (sweep BYO path; reference path unaffected). Drafted as a
   prepend in `CHANGELOG_paddelay_scope_entry.md` alongside this finding.

2. **Commit this finding and the CHANGELOG update at repo root** on
   `feature/device-calibrated-noise`. No code changes accompany this commit.

3. **Next live work item: run D3.5 gate-2.** Execute the q10 BYO sweep at
   production scale (40 seeds × 60 kicks × 1000 shots × calibration
   `08c3c70f`) via `examples/byo/floquet_dtc_q10_sweep.yaml`, aggregate the
   device-calibrated arm into `aggregated_autocorr.dat`, and compare to the
   unchanged committed reference. Expected outcome: convergence within
   per-kick sem tolerance, demonstrating that the fixed sweep BYO path
   reproduces the reference path's idle-decoherence physics within statistical
   error. Budget ≥4h (post-fix Kraus pipeline is ~48× slower per instance than
   noiseless on this scale; BYO machinery's per-placement transpile overhead
   adds additional headroom).
