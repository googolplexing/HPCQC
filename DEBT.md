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
