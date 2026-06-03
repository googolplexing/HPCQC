# HPCQC technical-debt ledger (feature/device-calibrated-noise)

Tracked, deferred items from the device-calibrated noise work (May 2026). Each
has an owner-decision, a clear trigger ("when this must be resolved"), and a
resolution plan, so nothing rots into an untracked loose thread. Close items by
deleting them here in the same commit that resolves them.

---

## D1 — Retire floquet_runner v1; promote v2
**Status:** deferred (intentionally).
**Why deferred:** v1 (`floquet_runner.py`, repo root) is kept ONLY as the A/B
reference for the prepare() seam. v2 has been confirmed bit-exact to v1
(`ab_compare --exact`, PASS, 2026-05-24).
**Trigger to resolve:** after the first full noiseless-vs-device-calibrated
physics run is banked and looks good (so the A/B reference is no longer needed).
**Plan:** `git rm floquet_runner.py`; rename `floquet_runner_v2.py` ->
`floquet_runner.py`. This also deletes the only real code duplication: v1's
inline `_prepare_device_circuits` (now living in `backends/prepare.py`) and v1's
now-unused top-level qiskit imports.

## D2 — build_noise_model T2>T1 Kraus: crash mitigated, audit-warning gap open
**Status:** crash UNREACHABLE in-tree (clamp present at every thermal site);
open only for the silent-clamp audit/warning gap. (Re-scoped per
RED-REVIEW-SPEC-002-7.5 §3.2 and BLUE-RESP-F5A-LOCUS-v1.0 §5.)
**What:** a qubit with T2 > 2*T1 would make `thermal_relaxation_error` produce
an invalid/empty Kraus that crashes under statevector ("QuantumError: Kraus is
empty"). A qubit with T1 < T2 <= 2*T1 produces a *valid but genuinely
non-unitary* Kraus (e.g. QB35 in the committed calibration).
**Clamp present — crash cannot occur:** all three thermal sites clamp
`t2 <= 2*t1` before calling `thermal_relaxation_error`:
  - `backends/noise_model.py:223-224` (`if t2_ns > 2*t1_ns: t2_ns = 2*t1_ns`),
  - `sweep/twin_simulator.py:167-168` (identical, in `build_placement_noise_model`),
  - `backends/device_noise.py` `_clamp_t2` (`min(t2, 2*t1)`, :72-91).
The committed calibration has ZERO qubits with T2 > 2*T1, so the clamp is a
no-op today; the one T1<T2<=2*T1 qubit (QB35) is handled on the statevector
path by `device_noise`'s resident-Kraus + `method="statevector"` pin
(`backends/prepare.py:379-390`), not by the clamp.
**Consumers (corrected):** `build_noise_model`'s only runtime consumer is
`backends/aer_gpu.py:137` (density_matrix; its config guard at :344-351 REFUSES
statevector with active noise). `sweep/twin_simulator.py` does NOT consume
`build_noise_model` — it has its own placement-aware `build_placement_noise_model`.
The device-calibrated path does NOT touch `build_noise_model` at all (it uses
`device_noise` via `prepare_simulation`).
**Trigger to resolve (the remaining gap):** a NEW consumer points
`build_noise_model` at `method="statevector"`. No current consumer does; the
device-calibrated statevector path is `device_noise`, already safe.
**Plan (D3.1 warning half, not yet landed):** make the three clamps log a
warning naming the qubit and original/clamped T2 when they fire, so a future
T2>2*T1 calibration is audited rather than silently corrected. The clamp itself
needs no change. Close D2 when the warning lands (or when `build_noise_model`'s
consumers converge onto `device_noise`).

## D3 — Reconcile the two noise systems (Stage 2 of sweep integration)
**Status:** pending (the core of sweep-engine integration).
**What:** there are two parallel noise vocabularies:
  - OLD: `backends/noise_model.py` (`build_noise_model`) + `sweep/noise_configs.py`
    (`NoiseConfig`): split t1_relaxation/t2_dephasing channels, logical circuits,
    density_matrix, no native decomposition / no scheduling / no real idle.
  - NEW: `backends/device_noise.py` + `backends/noise_spec.py` (`NoiseSpec`):
    combined thermal channel, native + routed + ALAP-scheduled circuits,
    statevector, real idle decoherence.
**Plan:** add a `source` axis to NoiseConfig; `source="channels"` keeps all 11
existing environments byte-identical (RED-SPEC-002 preserved), `source=
"device_calibrated"` routes through `backends/prepare.prepare_simulation`,
bypasses the circuit packer, and runs per-circuit statevector. Document the
vocabulary mapping explicitly. Resolving D3 is the natural place to also resolve
D2.
**Progress:** F5a per-placement composition (D3a) device-layer seam landed
(D3.2, commit `26d0beb`, verified job 18810719). Remaining: route
`device_calibrated` via the `source` axis (D3.3), the BYO counts execution path
(D3.4), and the gate-2 reproduction (D3.5).

## D3a — Per-placement device-calibrated noise composition (F5a)
**Status:** device-layer seam DELIVERED (D3.2, commit `26d0beb`); engine wiring
+ interim guardrail OPEN. (Required by RED-REVIEW-SPEC-002-7.5 §3.2 / §4 Q5.)
**What:** `prepare_simulation` and the device-cal builders
(`device_noise.build_control_readout_noise_model` :198,
`build_relaxation_pass` :377) historically self-selected qubits from
`num_qubits` alone, so a multi-placement `device_calibrated` run would apply one
noise model to every placement — voiding the placement dimension.
**Delivered (D3.2):** optional `physical_qubits`/`physical_edges` thread the
placement's qubits into BOTH builders (the desync-critical pair, since each
self-selects independently) via `noise_model._resolve_selected`, plus an
identity `initial_layout` so the routed circuit lines up with the
placement-keyed noise; `physical_qubits=None` preserves self-selection
byte-identical (F4 baseline). Fail-loud asserts on len/name/edge. Verified
in-container (`tests/unit/test_f5a_placement_noise.py`, 13/13). The
density_matrix analogue already shipped as
`sweep/twin_simulator.build_placement_noise_model` (the reference template).
**Interim mitigation (RED-ACCEPTANCE §2), still required until D3.4:** the
engine does not yet pass placements into `prepare_simulation`, so a routed
`device_calibrated` run (after D3.3, before D3.4) is still placement-blind.
Until D3.4 wires the call site, restrict `device_calibrated` to a single
placement (`top_1`) OR stamp the Parquet flag `noise_placement_independent=true`
(recording the single `physical_qubit_set` used), so a placement-blind run is
never read as placement-resolved.
**Trigger to close:** D3.4 wires the engine to pass each placement's
`physical_qubits`/`physical_edges`, and gate-2 (D3.5) verifies index-k idle
relaxation maps to `physical_qubits[k]` on the scheduled circuit. Then drop the
guardrail and close D3a.

## D4 — Multi-node ENSEMBLE scatter for large campaigns (> 1 node of instances)
**Status:** partially exists — important distinction below.
**Two kinds of parallelism, do not conflate:**
  - Kind 1 (EXISTS): distributing ONE large simulation across nodes. `aer_gpu.py`
    enables Aer/cuStateVec blocking (`blocking_enable`, `blocking_qubits=29`) for
    `num_qubits >= 30`, sharding a single statevector too big for one GPU across
    GPUs/nodes. Multi-node `srun` launch is tested (tests/test_fh_2x3_12q.sh,
    test_qaoa_12q.sh, ... print SLURM_NNODES). CPU path has blocking off
    (aer_cpu.py blocking_qubits=0). This is for big-n circuits, not for ensembles.
  - Kind 2 (MISSING): distributing MANY independent instances/sweep-points across
    nodes. Both floquet runners AND `sweep/eval_runner.py` /
    `sweep/execution_planner.py` fan out with `multiprocessing.Pool` -- SINGLE
    NODE. So > 128 independent instances run in waves on one node, not N-wide
    across nodes.
**What's needed:** for embarrassingly-parallel ensembles (e.g. 512 instances on
4 nodes), add an ensemble-level multi-node dispatch: SLURM job array or
`srun -N<k>` with per-task instance assignment via SLURM_PROCID (the launch
pattern the GPU test scripts already use), partitioning instances/sweep-points
across tasks; each task runs a single-node Pool over its chunk. The runners'
`--single-instance-id` flag is the building block. No MPI required for this.
**Interaction to design for:** Kind 1 and Kind 2 compete for the same nodes; a
campaign mixing large-n circuits (want Kind 1) and many small instances (want
Kind 2) needs the planner/execution layer to choose per-workload.
**Trigger:** when a campaign needs more concurrent independent instances than one
node's core count (128 on LUMI standard).

## D5 — Stale SLURM helper scripts
**Status:** minor.
**What:** `slurm_floquet_debug.sh` uses a stale `--backend` flag and
`slurm_floquet_parallel.sh` uses an old positional `q50-noise` interface; both
predate the current `--noise-source` CLI and would fail against today's runner.
**Plan:** update to the current CLI or remove. Not logical-gates related.

## D6 — Synthetic-tier (channels) BYO counts mode
**Status:** deferred (tracked, per RED-REVIEW-SPEC-002-7.5 §4 Q4).
**What:** the 11 synthetic `NoiseConfig` tiers are expectation-value (⟨H⟩)
oriented under density_matrix. BYO circuits use a counts→autocorrelator
observable. A counts mode for the synthetic tiers is a separate increment;
`device_calibrated` + `noiseless` is sufficient for the §7.5.6 example and the
gate-2 reproduction.
**Trigger to resolve:** when a BYO experiment needs a synthetic density_matrix
tier (not `device_calibrated`/`noiseless`) under the counts observable.
**Plan:** give the synthetic-tier path a counts mode parallel to the
`device_calibrated` counts path built in D3.4, sharing the same
counts→autocorrelator helper.

## D7 — Echo circuit + multi-observable BYO factory contract (D3.6)
**Status:** deferred (tracked; surfaced by the researcher's `Floquet_DTC_AK10_echo.py`).
**What:** the current BYO contract (D3.4) is single-circuit / single-observable:
one factory builds ONE circuit per grid point, and the engine computes ONE
counts→autocorrelator per (seed, placement, env). The researcher's going-forward
script runs TWO circuit families per instance and TWO+ observables:
  - `build_autocorr_circuit` (forward `num_kicks` Floquet periods) → A(n) via
    `get_autocorrelation`;
  - `build_echo_circuit` (forward `num_kicks` + `num_kicks` CONJUGATE periods,
    a Loschmidt echo) → echo value `sqrt(abs(A_echo(n)))`;
  - a derived/normalized quantity `A(n) / A_0(n)` combining both.
  Also feature flags now parameters: `Jz_on`, `hz_on`, `Jz_frozen`/value,
  `Initial_state`, `backend_code` (maps to noise source).
**Why deferred (Path 1 decision):** gate-2 (F4 reproduction) is pinned to the
AK7-shaped bank (single forward circuit, single autocorrelator,
batched-per-instance, one seed) and does NOT need echo. The single-observable
path is a strict SUBSET of the echo path, so D3.4b/c built for gate-2 are reused
verbatim when echo lands — nothing is wasted. Echo is a genuine contract
extension (multi-circuit-family declaration, configurable observable, cross-grid
check + placement over two families) that deserves its own design note, not a
rushed bolt-on into the gate-2 path.
**Trigger to resolve:** after gate-2 (D3.5) is signed off, and when the
researcher's standard workflow is the echo/normalized analysis. Confirm whether
echo is their everyday tool (→ D3.6 is the main event) or occasional.
**Plan (D3.6):** extend the factory contract so a factory may declare multiple
named circuit families per grid point (e.g. `{"autocorr": qc1, "echo": qc2}`),
and make the observable configurable per family (`autocorrelator`,
`echo=sqrt|A|`, derived `ratio=autocorr/echo`) rather than hardwiring
`get_autocorrelation`. Reuse the D3.4 counts worker, seeding (one per
instance), batching (one run per instance over the kick-list), and placement
unchanged — echo extends, does not replace. New design note: BLUE-DESIGN-D3.6.
The observable formula is researcher-edited (AK7 vs AK10 differ in
`get_autocorrelation`'s plus/minus form — mathematically identical, but it
confirms the observable must be configurable, not frozen).

## D8 — Hazard class: third-party-library precondition silently violated by an incomplete Target basis
**Status:** specific instance RESOLVED in this commit; ledger entry RETAINED
as a hazard-class record so the next instance of the pattern catches at the
assertion, not at the autocorrelator.

**Class.** A noise pass declared with `op_types=[X]` only fires when the
scheduled DAG actually contains X-typed instructions. For variable-duration
idle steps, that requires the upstream scheduling pad pass to be able to
insert X — which it gates on a Target-supported-instruction check
(`Target.instruction_supported("delay", qargs=(q,))` for PadDelay). If the
HPCQC-built Target lacks X in its supported-instruction set, the pad pass
silently skips insertion on every qubit and the noise pass is starved.
There is no error and no default-level warning trail (PadDelay's silent-skip
branch is a `logger.debug` only). Inspection-level review of the
HPCQC-internal chain (`_resolve_selected`, `initial_layout`,
`op_types=[Delay]`) can pass while the bug is fully active.

**Concrete instance (this commit fixes it):** the committed
`_NATIVE_BASIS = ["r","rz","sx","x","cz","id","measure"]` in
`backends/prepare.py` omitted `"delay"`, so the device-calibrated Target
did not list delay among its supported instructions. PadDelay therefore
silently skipped Delay insertion across every device-calibrated run on
this branch, and the `RelaxationNoisePass(op_types=[Delay])` registered
by `build_relaxation_pass` (`backends/device_noise.py`) had no Delays to
act on. The variable idle-time decoherence component documented in the
prepare/device_noise docstrings was silently inactive in production from
the day device_noise landed until this commit.

**Caught by:** the D3.2 §2.1 integration test
`test_idle_relaxation_tracks_placement_through_full_schedule`. All three
worker-chain variants (`id`, `sx`, `sx`+`barrier(1)`) produced statistically
indistinguishable target survival (~0.87 across 4096 shots, 0.3σ spread),
back-solving to gate-time-only |1> exposure (~140-170 ns), independent of
the worker chain's actual 5 µs duration. The integration test was the
first one designed to traverse the production scheduler+padding+relaxation
path; the existing direct-Delay tests bypassed PadDelay by running
explicit-Delay circuits on `prep.simulator` directly, which is why they
passed despite the underlying defect. See
`FINDING-PADDELAY-IDLE-NOT-INSERTED-v1_0.md` for the full code-path trace,
upstream Qiskit cross-check, and empirical reconciliation arithmetic.

**Mitigations in this commit (defense in depth):**
  - `prepare.py:50` — `"delay"` added to `_NATIVE_BASIS` (the structural
    fix).
  - `prepare.py`, immediately after `Target.from_configuration(...)` —
    runtime precondition assertion: raises `RuntimeError` if `"delay"` is
    absent from `target.operation_names`. Converts the implicit upstream
    library precondition into an explicit in-repo runtime invariant. If
    a future edit drops `"delay"` from `_NATIVE_BASIS` (or otherwise fails
    to register it), the prepare pipeline fails loud at Target
    construction instead of silently producing noise-deficient results.
  - `tests/unit/test_f5a_placement_noise.py` — new structural regression
    guard `test_prepare_simulation_inserts_delays_in_scheduled_circuit`:
    asserts that at least one `Delay` instruction survives into the
    scheduled circuit emitted by `prepare_simulation` on a multi-gate
    input. Catches any future regression that re-introduces the silent
    skip via a different route (e.g. a passmgr config that strips delays
    post-PadDelay, or a Target construction path that bypasses
    `_NATIVE_BASIS`).
  - `tests/unit/test_f5a_placement_noise.py` — strengthened docstring on
    `test_idle_relaxation_tracks_placement_through_full_schedule`
    spelling out that the swap differential is the §2.1 index-alignment
    proof, not just a magnitude band.

**Supersession of pre-fix artifacts (scope corrected — see
`FINDING-PADDELAY-SCOPE-v1_0.md`).** The blast radius first recorded here was
over-broad. Two architecturally disjoint idle-decoherence pipelines exist: the
reference path (`floquet_runner.py:_prepare_device_circuits` +
`device_noise.build_relaxation_pass`, an Aer-side `NoiseModel` custom pass) and
the sweep BYO path (`prepare.prepare_simulation` -> `PadDelay` +
`RelaxationNoisePass`). The `_NATIVE_BASIS`/`PadDelay` defect lived ONLY in the
sweep BYO path; the reference path never consumed `prepare.py` and never had
the bug. Empirically confirmed: a post-fix device-calibrated re-baseline
(job 18874828, same config `master_seed=0, initial_state=3, 40 instances,
calibration 08c3c70f`) reproduced
`examples/reference/floquet_dtc_q10_device-cal_agg.dat` byte-identically
(SHA256 `aa7084f2...`). Corrected scope:
  - NOT invalidated (runner-produced, unaffected by the bug):
    `examples/reference/floquet_dtc_q10_device-cal_agg.dat` and the
    `device_cal_mean / device_cal_sem` columns of
    `examples/reference/floquet_dtc_q10_autocorr.csv`. These remain the
    authoritative D3.5 gate-2 anchor; NO re-baseline is required.
  - Would be invalidated (under-decohered) ONLY if produced through the sweep
    BYO path pre-fix -- i.e. any such `/byo` HDF5 device-calibrated
    autocorrelator. No sweep-BYO-produced device-calibrated artifacts are
    committed, so nothing requires re-baselining in practice.
The pre-fix tag `pre-paddelay-fix` still preserves the broken-regime tree for
diagnostic comparison of the sweep BYO path. D3.5 gate-2 is therefore a
meaningful test of the fix: it compares the post-fix sweep BYO aggregate
against the unchanged runner-produced reference, and convergence within
per-kick sem is the success criterion (not a re-baseline gate).
Unaffected (verified by code-path inspection): the F4 noiseless baseline
(`floquet_dtc_q10_noiseless_agg.dat`, `noiseless_mean/sem` columns) routes
through `_prepare_noiseless`, which bypasses `_NATIVE_BASIS`, the Target,
and PadDelay entirely; resident gate-time relaxation on `r/sx/x/cz` and
the depolarizing/readout channels fire on gate ops and are unaffected
by the Delay-insertion defect.

**Pattern for future D3.x work.** When an HPCQC noise pass depends on a
downstream-library precondition (this case: RelaxationNoisePass → PadDelay
→ Target supports `delay`), convert the precondition into an in-repo
runtime assertion at the construction site. The
`if "delay" not in target.operation_names: raise RuntimeError(...)` in
`_prepare_device_calibrated` is the canonical example. Future passes
should follow the same pattern: any new `op_types=[OtherOp]` registration
should pair with an assertion that `OtherOp.__name__` is in the relevant
Target's `operation_names` at the prepare step. The lesson banked is that
a structurally correct in-repo chain proves nothing if one of its
preconditions is an implicit assumption about downstream-library behavior;
preconditions must be made explicit invariants.

**Logging follow-up (low priority).** PadDelay's silent-skip path emits
only `logger.debug` ("No padding on qubit %d as delay is not supported on
it"), which is filtered out at default log levels. Worth setting
Qiskit's `qiskit.transpiler.passes.scheduling` logger to INFO inside the
container for acceptance/diagnostic runs, so the trace surfaces in slurm
logs. Not gate-blocking; track here for the next general logging pass.

## D9 - Hazard class: two disjoint pipelines apply the same idle-decoherence physics through different code
**Status:** open (tracked; not blocking). Surfaced by
`FINDING-PADDELAY-SCOPE-v1_0.md` while explaining why the PadDelay fix left the
committed reference byte-identical.

**What.** Idle-time thermal relaxation is implemented twice, on two paths that
share no code:
  - reference path -- `floquet_runner.py:_prepare_device_circuits` does its own
    transpile + ALAP scheduling and attaches relaxation via
    `device_noise.build_relaxation_pass`, registered as an Aer-side
    `NoiseModel` custom noise pass (runs at simulate time on any circuit that
    already contains Delays);
  - sweep BYO path -- `prepare.prepare_simulation` builds a device-calibrated
    `Target` from `_NATIVE_BASIS`, schedules with Qiskit `PadDelay`, and applies
    relaxation with `RelaxationNoisePass(op_types=[Delay])`.
They use different scheduling mechanisms, different decoherence-application
mechanisms, and do not share `_NATIVE_BASIS`.

**Why it is a hazard.** The two paths form an *unintended* consistency check.
A defect in one path's idle-decoherence (e.g. D8's `_NATIVE_BASIS`/`PadDelay`
silent skip) is invisible in the other path's output, so a bug can ship in one
while the other looks healthy. A divergence between them is ambiguous: it could
be a genuine physics-modeling difference or a regression in one path, and
nothing in the repo disambiguates which. Today the only place the two are
cross-checked is D3.5 gate-2 (sweep BYO aggregate vs runner-produced reference).

**Trigger to resolve.** After D3.5 gate-2 is signed off (the one live
cross-check between the paths), schedule a consolidation discussion. Open
question: consolidate onto a single shared scheduler + idle-decoherence
implementation used by both `floquet_runner.py` and the sweep BYO path, or keep
them separate and treat gate-2 as a deliberate periodic cross-check (if kept
separate, add an explicit recurring cross-check rather than relying on gate-2
incidentally).

**Not blocking** the echo workstream (D7) or gate-2 itself.

## W1.4-1 — Pool sized by the heavy (device-cal) arm for ALL units
**Status:** deferred (intentionally; RED-RESP-W1.3-VERIFY-AND-W1.4-CAP-RULINGS D2 conservative-now).
**What.** The W1.4 worker cap sizes the whole forkserver pool by the
`device_calibrated` per-unit peak and treats every unit as heavy, including the
lighter `noiseless` arm. The result under-packs noiseless-heavy sweeps (fewer
concurrent workers than memory would allow) but never OOMs. The footer records
`all units treated as heavy [device_calibrated], D2 conservative` so the
under-pack is auditable.
**Trigger to resolve.** When noiseless-arm wall-clock becomes a measured
bottleneck on a real sweep, or before any run where the two arms' per-unit peaks
differ enough that single-pool sizing wastes a non-trivial fraction of the node.
**Plan.** Split into two pools (or one pool with per-unit memory weights):
probe each arm's VmHWM separately (the probe already isolates the device-cal
arm; add a noiseless probe), then size each arm's concurrency against `safe_mem`
independently. Keep per-unit `run_one_unit` byte-identical so the canary
byte-match is unaffected.

## W1.4-2 — per_unit_peak does not measure the set_forkserver_preload delta
**Status:** open (tracked; RED-RESP D6-ii REJECT-for-now).
**What.** D6 rejected adding `set_forkserver_preload` in W1.4 (it would perturb
the proven forkserver topology that the 2-seed canary pins). The cap therefore
sizes each worker at its full standalone `VmHWM`, which is correct for the
current no-preload topology (NF5). We have not measured what a preloaded server
would change: the 40-way cgroup peak with vs without preload, and whether the
`.so`-page sharing already captures most of the benefit.
**Trigger to resolve.** When packing density becomes the binding constraint on
throughput (i.e. the memory term, not cores, repeatedly limits the cap on real
sweeps) and the extra concurrency would matter.
**Plan.** A dedicated measurement step (NOT on the canary path): run the 40-way
corpus with and without `set_forkserver_preload`, compare cgroup `memory.peak`,
and evaluate the same change in `floquet_runner`. If material, design the
preload as its own reviewed change with its own canary.

## W1.4-3 — per_unit_peak is a single-unit probe, not a marginal-RSS measurement
**Status:** open (tracked).
**What.** `per_unit_peak` is one device-cal unit's standalone `VmHWM`. Because
workers share `.so` pages (NF5), the *marginal* resident cost of the Nth
concurrent worker is below its standalone peak, so the cap is conservative (it
over-counts shared pages -> under-packs). Acceptable and safe now; imprecise.
**Trigger to resolve.** Same as W1.4-2 — when the memory term is the binding
constraint and density matters.
**Plan.** Add a sub-second cgroup `memory.peak` sampler + a per-worker PSS
(proportional set size) read during a calibration run to measure the true
marginal cost, and feed a marginal figure (not the standalone peak) into the cap
default. Pairs naturally with the W1.4-2 measurement.
## W1.5-OBS-1 — D1/D2 memory probe runs silently (no phase logging, non-standardized output)
**Status:** open (tracked).
**What.** The D1/D2 `per_unit_peak` probe (sweep_engine `_execute_byo_group`)
runs ONE device-calibrated unit alone in `Pool(1)` before dispatching the main
wave. For a q10 / 60-kick / 1000-shot unit that is ~12 min of wall during which
the engine emits NOTHING -- the only trace is the `[probe:device_calibrated_
VmHWM]` token folded into the post-probe `dispatching ...` line. From the
outside it reads as a multi-minute hang at "1 manual placement(s)". The
probe / dispatch / aggregate phases also use ad-hoc line formatting rather than
one grammar.
**Trigger to resolve.** Any time a human watches a device-calibrated BYO run
(every gate / campaign run); first surfaced live during the W1.6 gate
(job 18958015).
**Plan.** (1) Emit a `── BYO memory probe (D1/D2) ──` section BEFORE the
`Pool(1)` runs: which unit (seed, qubits, kicks, shots), why (binding heavy
arm), that the main pool is deferred until it completes (~one unit wall), and
that the probe result is kept (reused as a real seed, not discarded). (2) A
standardized completion line: `probe complete: per_unit_peak=X GiB (VmHWM) |
elapsed=Ns | kept seed=S`, then the existing dispatch line under a
`── BYO dispatch ──` header. (3) Factor a single phase-line formatter
(`key=value` grammar) reused across probe / dispatch / aggregate so the output
is uniform. Observability only; does NOT change the cap math or any physics.
## PLACEMENT-1 — researcher cannot select physical qubits (replication is broken for the device-calibrated arm)
**Status:** Phase 1 DONE — researcher-controlled placement ADOPTED per
RED-RESP-GATE2-FAILURE-RECONCILIATION-AND-PLACEMENT-CONTROL-RULING §6 (conditions
1–4 verified at `1bbe5da`); the `physical_qubits` seam is committed (`d3f1b3b`),
18/18 unit, confirmed on real deps (job 18946498). It is the mechanism for the
gate-2 reconciliation (Step-1 confirmatory + Option-1 reference pin), now in
progress. **Phase 2 (QPU integration) DEFERRED** to its own design + review
(RED-RESP §6).
**What.** A BYO sweep's physical qubits are an *output* of the placement solver
(`_execute_byo_group` always calls `self._solver.find_all_placements(...,
strategy="max_fidelity")`); there is no YAML seam to supply them. A
device-calibrated noise model is determined by two inputs — the calibration
snapshot (selectable via `calibrations:`) and the physical qubits (NOT
selectable). The asymmetry breaks replication: to reproduce a prior experiment
one must pin both, but the solver picks current-optimal qubits that in general
differ from a past run's. Gate-2 (job 18943612) is the live proof — the
solver's `top_1` (`QB11…QB26`) ≠ the runner reference's fidelity self-selection
(`QB1,2,5,6,7,9,10,11,12,13`); a ~2.4× T2 difference on the swapped qubits drove
a 52/60-kick > 5σ divergence (see BLUE-FINDING-W1_6-GATE2-PLACEMENT-PROVENANCE).
**Trigger to resolve.** Before any replication of a placement-keyed experiment,
and as the mechanism for the gate-2 reconciliation (findings doc §7 option 2 /
§6 confirmatory run). Phase 1 (simulation) is the near-term, gate-2-unblocking
increment.
**Plan.** Phase 1 (sim, small): add a YAML `physical_qubits:` field accepting a
list of placements (each a list of qubit strings); when present, bypass the
solver and feed the lists to the existing `_resolve_selected` placement path;
extend its single-placement fail-loud validation (count==num_qubits, names in
calibration, real calibrated edges) list-wise. Task multiplication and the W1
cap parallelism already handle the rest. Phase 2 (QPU, larger, separate review):
integrate manual/solver placements through the existing-but-unwired `MixedPacker`
(qubit+edge overlap → shared rounds; overlap → separate circuits) and the
`iqm_qpu` backend (`VTT_BATCH_LIMIT`, ≤100/batch). BYO is simulation-only today;
Phase 2 is integration, not plumbing.


## REPRO-PLACEMENT-1 — device-calibrated transpile is unseeded and the resolved layout is unrecorded (free-layout references are not reproducible)
**Status:** open (tracked; RED ruling #4 in RED-RESP-STEP1-COLLAPSE-CANONICAL-PLACEMENT-AND-GATE-SEMANTICS-v1.0).
**What.** For a device-calibrated run with no pinned placement
(`physical_qubits=None`), `backends/prepare.py:_prepare_device_calibrated`
transpiles **free-layout** (`initial_layout=None`) with **no `seed_transpiler`**
(`prepare.py:357-364`; no seed anywhere on the device-calibrated path). Sabre's
layout — which physical qubit sits at which chain position — is therefore
**non-deterministic** run-to-run, and the **resolved layout is not recorded**:
the runner metadata persists circuit/simulator seeds only, explicitly not the
transpiler (`floquet_runner_v2.py:268-292`). So a free-layout device-calibrated
reference's exact placement is neither reproducible nor recoverable. This is the
W1.6 Step-1 blocker (see BLUE-FINDING-STEP1-PLACEMENT-NONDETERMINISM-AND-OPTION1-COLLAPSE-v1.0).
**Why it matters (scope).** This affects **every** device-calibrated reference
produced via the free-layout path, not just W1.6. The averaged autocorrelator
depends on which physical qubit occupies which chain position (per-qubit T1/T2
*and* chain-position dynamics), so a non-reproducible layout is a
non-reproducible reference.
**Why it's not biting the W1.6 gate.** The pinned-path Option-1 plan pins both
arms via `--physical-qubits` (`initial_layout=list(range(n))`), which sidesteps
free-layout for that gate. REPRO-PLACEMENT-1 is the fix for the *general*
(non-pinned) path.
**Trigger to resolve.** Before banking any new free-layout (non-pinned)
device-calibrated reference intended for reproduction or comparison.
**Plan.** (1) Set a fixed `seed_transpiler` on the device-calibrated transpile
(mirror the `orchestration/workflow.py:161` `seed_transpiler=42` precedent) so
the free-layout becomes deterministic. (2) — the real provenance, per Red —
**record the resolved logical→physical layout** in the run metadata. A seed only
reproduces given identical Qiskit/rustworkx versions; the recorded layout is
version-independent provenance and is what an auditor or a replay actually needs.
Optionally, replay/assert against the recorded layout on re-run.
**Guard until then.** Do not bank a free-layout device-calibrated reference for
later comparison without either pinning the placement (`--physical-qubits`) or
recording the resolved layout.

## D10 — BYO expected-group inventory generator (option (i), BYO half)
**Status:** deferred (intentionally; battery half landed).
**What:** option (i) closes the WHOLLY-ABSENT-group blind spot in the multi-node
merge by writing an expected-group inventory (`campaign_expected.json`) the merge
asserts the unioned group set against. The BATTERY generator shipped (the engine
enumerates battery groups via `battery_paths.battery_group_path` keyed by
`group_key_from_path`; the merge-side group-set assert in
`assert_complete_and_reduce(..., expected_groups=...)` is reducer-AGNOSTIC). The
BYO (`ByoAutocorrReducer`) generator is NOT written: the BYO group key carries an
`obs_tail` resolved at WRITE time (`byo_observable_subpath(...)` + the
`_byo_collision_stems` disambiguation, `hdf5_writer.py`:446-450), so predicting
the BYO key at setup needs that logic hoisted/replicated — unlike the clean
battery `group_path`.
**Why it's safe to defer (exposure asymmetry, per Red §2):** a wholly-absent BYO
group only escapes the existing short-count guard at `num_seeds == 1` (the merge
sees no other seed to make the group "short"). The echo campaign is multi-seed
(`num_seeds >= 2`), so it stays protected by the short-count guard. The merge CLI
lists only `BatteryReducer` in `_REDUCERS_WITH_INVENTORY`, so a BYO merge skips
the group-set check (no false failure) — and a multi-seed BYO merge does not need
it.
**Trigger to resolve:** before a SINGLE-SEED BYO multi-node campaign banks
results. (Multi-seed BYO multi-node may bank now; battery multi-node may bank
once the (i-b) gate is green.)
**Plan:** (1) hoist the BYO write-time obs/collision logic into a pure
`byo_group_path(...)` + `byo_group_key_from_path(...)` pair in a stdlib-only
module (mirroring `battery_paths.py`), with the SAME single-source-of-truth
discipline (the writer delegates; `ByoAutocorrReducer.extract` parses via the
shared parser; the inventory builds + keys via the same pair). (2) accumulate the
BYO expected set in the engine's BYO execute path (before the shard slice) and
add `"ByoAutocorrReducer"` to `_REDUCERS_WITH_INVENTORY`. (3) add a BYO analog of
`tests/fanout_negative_dropgroup.py` (i-b) and an offline round-trip test that
includes a collision-disambiguated obs_tail (the BYO drift point).
**Guard until then:** do NOT bank a single-seed BYO multi-node campaign. The
launcher banner and the merge CLI docstring both carry this prohibition.

## D11 — Regenerate the expected-group inventory on a manifest-resume (shard mode)
**Status:** deferred (intentionally; fail-loud-safe, per RED-ACCEPT-OPTION-I-AND-BATTERY-MULTINODE-LIFT §3 flag 2).
**What:** the engine writes `campaign_expected.json` only on a FRESH shard run
(`self._manifest_fresh`). A manifest-RESUME (manifest present at start) executes
only the remaining tasks, so it does NOT regenerate the inventory — by design, so
a resume cannot clobber the authoritative fresh inventory with a partial set.
**Why it's safe (not a correctness bug):** both states are safe — (a) a fresh run
that finished wrote the full inventory → the merge group-set-checks it; (b) a
fresh run that CRASHED before finalization left no inventory, and a subsequent
resume writes none either → a multi-rank battery merge then FAILS LOUD (Q1,
inventory-required), forcing a fresh re-run. Fail-loud, never silent-wrong.
**Cost (usability only):** a crashed-then-resumed multi-rank battery campaign
cannot be group-set-merged until it is re-run fresh.
**Trigger to resolve:** when a long multi-rank battery campaign is expensive
enough that re-running fresh after a mid-run crash (rather than resuming) is a
real cost.
**Plan:** on a resume in shard mode, rebuild the expected set from the FULL task
list (before the manifest resume-filter at sweep_engine.py ~1628) — enumerate the
inventory independent of which tasks still need executing — and write
`campaign_expected.json` (idempotent: the full set is identical regardless of how
many tasks remain). Then a resumed campaign is group-set-mergeable.
**Guard until then:** if a multi-rank battery campaign crashes mid-run, re-run it
FRESH (clear campaign_manifest_rank*.json / the output dir) rather than resuming,
when the group-set merge is required.
