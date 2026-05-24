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

## D2 — build_noise_model: latent T2>T1 empty-Kraus bug
**Status:** latent (NOT crashing today), tracked.
**What:** `backends/noise_model.py:build_noise_model` emits a genuine
non-unitary Kraus thermal channel for any qubit with T2 > T1 (e.g. QB21). Under
statevector that hits the "QuantumError: Kraus is empty" precompute gap (same
root cause as the device-calibrated bug fixed via `method="statevector"` pin in
`backends/prepare.py`).
**Why it's not biting now:** its consumers (`sweep/twin_simulator.py`,
`backends/aer_gpu.py`) run under **density_matrix**, where Aer enables the
superop method unconditionally -> the gap never triggers.
**Trigger to resolve:** BEFORE any consumer of build_noise_model runs it under
`method="statevector"` (the sweep engine will want this for n > ~12 qubits,
where density_matrix's O(4^n) becomes infeasible).
**Plan (part of D3):** give build_noise_model the same statevector-safety as
device_noise — either force the kraus precompute by pinning statevector where
it's used, or converge its consumers onto `device_noise` for the faithful path.
Decide during D3.
**Guard until then:** do not switch any build_noise_model consumer to
statevector without resolving this first.

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
