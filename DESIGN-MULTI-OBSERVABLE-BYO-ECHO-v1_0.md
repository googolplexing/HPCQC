# DESIGN — Multi-Observable BYO Sweep Path (Echo)

**Version:** 1.0 (design proposal — NOT yet implemented)
**Date:** 2026-05-27
**Branch target:** `feature/device-calibrated-noise`
**Base HEAD:** `f0ce463` (+ pending FINDING/CHANGELOG commit)
**DEBT item:** D7 — "D3.6 echo + multi-observable BYO factory"
**Spec anchor:** SPEC-002 §7.5 (BYO factory contract)
**Researcher source:** `Floquet_DTC_AK10_echo.py`
**Factory (built, verified):** `examples/byo/floquet_dtc_echo.py`

---

## 0. Why this is a design doc and not a patch

The current BYO sweep path is single-circuit-per-task end to end:

- `circuit_loader._load_script` calls `fn(**params) -> QuantumCircuit` — one
  circuit.
- `_analyze` extracts one connectivity graph; the placement solver runs on
  that one graph.
- `_execute_byo_group` produces `byo_results` as one autocorrelator series per
  (seed, placement, env); `writer.write_byo_result` writes one record each.
- The `.dat` aggregation under `_byo_dat_dir` is one mean+sem series per
  (placement, env).

The echo experiment is intrinsically **two circuits + one derived ratio** per
(seed, num_kicks) point:

1. autocorrelator circuit `A(0)A(T)` — `build_circuit`
2. echo circuit `A_0` — `build_circuit_echo` (forward + conjugate, doubled
   depth)
3. derived `A(0)A(T) / A_0` where `A_0 = sqrt(|autocorrelator-of-echo-circuit|)`

The ratio cannot be computed in the factory — it needs measured counts from
both circuits, which exist only post-simulation. So the engine must (a) build
and simulate two circuits per task, (b) carry both results through WAL/HDF5,
and (c) compute the ratio in the analysis/aggregation stage. That is a change
to the core task model, which per our working rules gets a reviewed design
before code.

---

## 1. Design constraints (carried from project rules + RED conventions)

1. **No team-name refs** in code, commit messages, or new filenames. Phase/spec
   IDs (D3.6, §7.5) are fine.
2. **Backward compatibility.** The single-observable BYO path
   (`floquet_dtc.py` / `floquet_byo_sweep.yaml` / the banked q10 sweep) MUST
   continue to work byte-identically. The echo path is additive.
3. **Resume-safe.** Anything the sweep produces must be re-runnable from clean;
   resume detection reads task completion from HDF5 reality (task IDs), not
   from SweepResult counters. The two-circuit task must mark complete only when
   BOTH circuits' results are durably written.
4. **Native-basis lowering is per-arm and downstream.** The factory stays in
   logical rx/rz/rzz. device_calibrated transpiles to prx/cz with calibrated
   durations + ALAP + PadDelay (the 90d329d path); noiseless runs logical on
   statevector. This is unchanged — both echo circuits flow through the same
   per-arm machinery as the single-observable circuit does today.
5. **Staged sub-steps over a monolith.** This design decomposes into 4
   independently-reviewable+verifiable increments (§5).

---

## 2. The echo physics (for reviewers)

Forward Floquet period P = rx(h_x) · rz(hz) · rzz(Jzz) on the chain, identical
every kick (time-independent drive). Autocorrelator circuit applies P^k then
measures; `A(0)A(T)` is the polarization autocorrelator (the existing
observable). Echo circuit applies P^k then (P†)^k then measures; under perfect
(noiseless) reversal P†P = I so the echo circuit returns the initial state and
its measured autocorrelator ≈ 1 at every kick (up to readout). Under noise the
echo circuit's autocorrelator decays — that decay is the pure-decoherence
envelope, since the dynamics cancel. `A_0 ≡ sqrt(|echo-autocorrelator|)`: the
sqrt accounts for the echo circuit being ~2× the depth (forward+conjugate), so
A_0 estimates the single-direction decoherence factor. The headline signal
`A(0)A(T) / A_0` divides the decoherence envelope out of the raw
autocorrelator, isolating the genuine DTC response from noise-induced decay.

Verified property of the factory (`examples/byo/floquet_dtc_echo.py`): the
conjugate half is the exact dagger U† of the forward half — reverse block order
(rzz→rz→rx) with negated angles; intra-block gates all commute (diagonal
rz/rzz; disjoint single-qubit rx), so AK10's not-fully-reversed within-block
iteration is the same unitary. Echo is mathematically exact noiseless.

---

## 3. Task-model options (the core decision)

### Option A — Two observables, one task, dual circuit build (RECOMMENDED)

Extend the BYO task to carry a list of `(observable_name, circuit_function)`
pairs instead of a single `circuit_function`. The engine builds both circuits
per (seed, num_kicks), simulates both (same placement, same noise env, same
shots), writes both raw autocorrelator series to HDF5 under
`.../obs=autocorr` and `.../obs=echo` subgroups, and the aggregation stage
computes the ratio series as a third derived `.dat`.

- **Placement:** both circuits have the SAME connectivity graph (the echo
  circuit is the autocorr circuit's gate set, doubled — same edges, same
  qubits). So the placement solver runs ONCE on the autocorr connectivity and
  the placement applies to both. No placement-model change. (Assert
  connectivity equality at build time as a guard.)
- **Counts/accounting:** one task now does 2× simulations. The counter fix at
  f0ce463 increments `total_simulations += len(byo_results)`; with two
  observables `byo_results` naturally carries both, so the counter stays
  correct without further change — it counts simulator runs, and there are now
  two per (seed, placement, env).
- **Resume:** task completion = both observable subgroups present in HDF5.
- **Pros:** one placement solve; shared transpile of the common sub-circuit is
  possible later; the ratio is a clean post-agg derivation; backward compatible
  (single-observable experiments declare one observable and behave exactly as
  now).
- **Cons:** touches the task dataclass, `_execute_byo_group`, the WAL line
  format (must carry observable name), `write_byo_result`, and the aggregation
  walker. The WAL format change is the riskiest bit (resume/recover parsing).

### Option B — Two separate experiments, joined in post

Declare `autocorr` and `echo` as two independent `byo_circuit` experiments in
the same YAML (different `circuit_function`, same everything else), let the
existing single-observable path run both, then compute the ratio in a separate
offline join keyed on (seed, num_kicks, placement, env).

- **Pros:** ZERO engine changes — uses the path exactly as-is. Lowest risk.
  Ships today.
- **Cons:** two placement solves (redundant — identical graphs); the join is an
  external script the engine doesn't know about (the ratio isn't a first-class
  sweep artifact); two HDF5 task families to keep aligned; easy to get the
  seed/placement pairing subtly wrong in the offline join. Doesn't satisfy the
  spirit of DEBT D7 ("multi-observable BYO factory") — it's a workaround.

### Option C — Echo-aware factory returns both, engine unpacks a tuple

Change the factory contract to allow returning `dict[str, QuantumCircuit]`
instead of a single circuit; engine detects the dict and treats each entry as
an observable.

- **Pros:** factory expresses the multi-observable intent directly; one call
  site.
- **Cons:** breaks the `fn(**params) -> QuantumCircuit` contract that
  `_load_script` + `_analyze` + the §7.5.1 pre-submit signature check all
  assume; every caller must handle both return types. More invasive to the
  loader than Option A is to the engine. Higher blast radius.

**Recommendation: Option A.** It localizes the change to the BYO execution +
storage path (where the f0ce463 counter work already lives), keeps the factory
contract intact (one function = one circuit, just declare two functions),
solves placement once, and makes the ratio a first-class artifact. Option B is
the fallback if engine-change risk must be deferred. Option C is not worth the
loader blast radius.

---

## 4. Option A — concrete shape

### 4.1 YAML surface (additive field `observables`)

```yaml
sweep:
  experiments:
    - type: byo_circuit
      label: floquet_dtc_echo
      circuit_script: examples/byo/floquet_dtc_echo.py
      # NEW: list of (name, function). When present, supersedes the single
      # circuit_function. When absent, behavior is exactly as today
      # (circuit_function defaults to build_circuit, one observable named
      # "default").
      observables:
        - {name: autocorr, function: build_circuit}
        - {name: echo,     function: build_circuit_echo}
      # NEW: derived observables computed in aggregation from raw ones.
      derived:
        - {name: ratio, expr: "autocorr / sqrt(abs(echo))"}
      fixed:
        num_qubits: 10
        epsilon: 0.03
      grid:
        num_kicks: {range: [0, 40]}     # AK10: 0..39
      disorder:
        source: file
        file: examples/byo/floquet_disorder_q10_echo.json
        initial_state: 3
      disorder_gates: [rz, rzz]
      seed_list: [0,1,2,3,4,5,6,7,8,9]  # AK10: num_gate_instances=10
      noise_configs: [noiseless, device_calibrated]
  calibrations:
    - examples/q50_calibration_20260524_08c3c70f.json
```

Backward-compat rule: if `observables` is absent, the engine synthesizes
`[{name: "default", function: <circuit_function>}]` and skips all derived/ratio
logic — the existing single-observable artifacts are produced unchanged.

### 4.2 Task dataclass

Add `observables: tuple[tuple[str,str], ...]` (name, function) and
`derived: tuple[tuple[str,str], ...]` (name, expr) to the BYO task. Default
both to the single-observable synthesis above. `circuit_function` stays for
back-compat (used to synthesize the default observable).

### 4.3 `_execute_byo_group`

Per (seed, placement, env), loop over observables: build each circuit (via
`load_factory(script, obs.function)`), simulate, collect the per-kick
autocorrelator series tagged with the observable name. `byo_results` entries
gain an `observable` field. Connectivity-equality guard: assert all observables
in an experiment share the autocorr's connectivity graph before solving
placement once.

### 4.4 WAL + HDF5

- WAL line gains an `observable` token (the D3.4c WAL already carries
  `group_path`; extend the byo-aware line and `verify_consistency` /
  `recover_from_wal` to round-trip the observable name). **This is the
  highest-risk increment** — resume/recover correctness.
- HDF5 layout: `.../<placement>/<env>/<observable>/autocorr` series. Existing
  single-observable runs write under `.../default/...` (one extra level), OR —
  to keep existing references byte-identical — keep the current layout when
  there is exactly one observable named "default" and only add the observable
  level when >1. (Prefer the latter to avoid touching the banked-reference
  layout.)

### 4.5 Aggregation + derived ratio

The `.dat` walker aggregates each raw observable to mean+sem as today, one
`.dat` per (placement, env, observable). Then for each `derived` entry it
evaluates the expr per kick over the aggregated raw series (numpy-safe eval of
a whitelisted expression — `autocorr`, `echo`, `sqrt`, `abs`, `/`, `*`, `+`,
`-`) and writes a derived `.dat`. The ratio plot AK10 draws
(`A(0)A(T)/A_0`) becomes `ratio.dat`.

Caveat to document: the ratio divides two shot-noisy quantities; at AK10's
100 shots × late kicks both shrink and the ratio's variance blows up. The
aggregation should propagate sem through the division (first-order: relative
variances add) and flag kicks where the echo denominator is within N·sem of
zero as unreliable.

---

## 5. Staged increments (each independently reviewable + verifiable)

0. **BYO `shots` config field (PREREQUISITE for AK10 fidelity).** As of HEAD
   `f0ce463`, `sweep_engine.py:2061` hardcodes `byo_shots = 1000` as the
   fallback shot count for envs whose own `shots == 0` (the noiseless arm).
   The inline comment claims it "is a config field" but it is NOT read from the
   experiment config — it's a literal. To replicate AK10's `num_shots=100`, the
   parser must read an experiment-level `shots` (the parser at lines 631–640
   currently reads `circuit_function`/`disorder_gates`/`seed_list` but not
   `shots` for BYO), thread it onto the BYO task/spec, and use it in place of
   the `byo_shots = 1000` literal. Small, localized, low-risk. Verify: q4 echo
   smoke with `shots: 100` actually runs 100 shots (check the per-result
   `"shots"` field written at line 2129). NOTE: this only governs the noiseless
   arm; the device_calibrated env carries its own `shots` (4096), which itself
   departs from AK10 (AerSimulator/noiseless only) — documented as expected,
   since device_calibrated is an ADDED arm, not part of AK10.

1. **Factory only** (DONE — `examples/byo/floquet_dtc_echo.py`, verified
   structurally: gate counts, exact-dagger property). Ship this first; it's
   inert until an experiment references it. Add a unit test asserting the
   echo circuit's conjugate is U† of the forward (multiset-per-block negation).

2. **Disorder JSON for q10 echo** — `examples/byo/floquet_disorder_q10_echo.json`
   matching AK10's RNG. **OPEN QUESTION (§6):** AK10 uses
   `np.random.default_rng(1234)` with `rng.uniform` per instance, NOT the
   `SeedSequence(0).spawn` / pcg64 mechanics of the existing q10 disorder file.
   The disorder file must reproduce AK10's exact angles to replicate AK10. See
   §6.

3. **Engine: observables list (build+simulate N circuits/task), HDF5 only,
   NO WAL change yet** — write each observable under its subgroup; gate the
   observable HDF5 level behind ">1 observable" to preserve single-observable
   layout. Verify: q4 echo smoke produces autocorr+echo series; single-obs q4
   smoke byte-identical to pre-change.

4. **Engine: WAL observable token + resume/recover** — the risky increment.
   Verify: kill+resume an echo sweep mid-run; confirm WAL replay reconstructs
   both observables; verify_consistency passes.

5. **Aggregation: derived ratio `.dat` + sem propagation + denominator-floor
   flagging.** Verify: ratio.dat matches an offline numpy recomputation from
   the raw aggregates.

Each increment is a separate commit with its own smoke verification on LUMI.

---

## 6. RESOLVED DECISIONS + DISORDER-PATH FINDINGS

User decisions (this session):
- **Standalone AK10 fidelity study.** The echo experiment is its own thing, not
  the D3.5 gate-2 comparison. Disorder uses AK10's exact `default_rng(1234)`
  draws, frozen into `floquet_disorder_q10_echo_ak10.json`.
- **Parameters:** 10 qubits, kicks 0..39, 10 seeds, ε=0.03, polarized init,
  100 shots, both `noiseless` (AK10's actual AerSimulator run) and
  `device_calibrated` (added native-basis noisy arm).
- **Disorder ranges/flags confirmed** against the existing q4 file: hz ≈ [-π,π],
  Jzz ≈ [-1.5π,-0.5π], all AK10 flags at default (random Jz/hz, no frozen
  coupling) — so the distribution already matched; only the exact draws differ
  by RNG mechanism.

Three findings about the disorder `generate` path at HEAD `f0ce463` (verified
against the tree; detailed in `PROCEDURE-SCRIPT-TO-BYO-SWEEP-v1_0.md` §4):

1. **`source: generate` is NOT engine-wired.** `resolve_disorder`'s `generate`
   branch needs a `sampler(rng, num_qubits) -> dict`, but the engine call site
   (`sweep_engine.py:430`) passes no `sampler`, so the branch raises. **`source:
   file` is the only functional disorder path** — the disorder JSON is mandatory
   for any BYO sweep today, not just AK10 replication.

2. **BYO `shots` field is inert** (= design increment 0). `sweep_engine.py:2061`
   hardcodes `byo_shots = 1000`; the experiment-level `shots:` is not read. AK10
   shot-count fidelity (100) requires wiring this.

3. **`generate` cannot reproduce AK10's RNG even once wired.** `_spawn_rng`
   offers `pcg64` (per-seed independent `SeedSequence.spawn` child) and
   `legacy_npr` (per-seed global reseed). AK10 uses a SINGLE serial
   `default_rng(1234)` stream. **A new generator option (proposed name
   `shared_default_rng`) must be added to `_spawn_rng`** in `byo_sweep.py` to
   replicate single-stream `default_rng` semantics — draw all instances from one
   `np.random.default_rng(master_seed)` advanced serially, in the factory's
   declared draw order. This is a prerequisite if `generate` is ever to match a
   `default_rng`-based researcher script without a frozen JSON.

---

## 7. Staged increments — UPDATED with the disorder-path work

The §5 increments still hold; add the disorder-path items:

- **Increment 0** (shots field) — as in §5; prerequisite for shot fidelity.
- **Increment 0b** (`generate` sampler wiring) — pass a `sampler` from the
  factory module (convention: factory exposes `sample_disorder(rng, num_qubits)
  -> dict`) into `resolve_disorder` at the engine call site, so `source:
  generate` works end-to-end. Verify: a `generate` sweep produces the same
  instances a frozen file would for the `pcg64` generator.
- **Increment 0c** (`shared_default_rng` generator) — add the single-stream
  `default_rng` option to `_spawn_rng` (Finding #3). Verify: generated angles
  byte-match `floquet_disorder_q10_echo_ak10.json` for seed 1234. With 0b+0c, a
  researcher could replicate AK10 via `source: generate` and DROP the frozen
  JSON — closing the "do I really have to make the JSON by hand?" gap properly.
- **Increments 1,3,4,5** — factory (done), engine HDF5 multi-obs, WAL, ratio
  aggregation, as in §5.

## 8. What ships when

- **Now:** factory (`floquet_dtc_echo.py`, verified) + this design doc + the
  frozen disorder JSON (`floquet_disorder_q10_echo_ak10.json`) + the echo sweep
  YAML + `PROCEDURE-SCRIPT-TO-BYO-SWEEP-v1_0.md`. All inert/safe until the engine
  increments land; the YAML's single-observable arm + Option B fallback run on
  today's engine (at 1000 shots until increment 0).
- **Then, staged:** increments 0 → 0b → 0c (disorder-path) and 3 → 4 → 5
  (multi-observable), each its own commit + LUMI smoke.
- **Gate-2 note:** this echo experiment is its own characterization study, NOT
  the D3.5 gate-2 comparison (the banked reference has no echo arm and uses
  60k/1000s/40seed, not AK10's 40k/100s/10inst). Keep them separate.
