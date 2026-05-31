# BLUE-REPORT — W1.6 Option-1 gate closure (PASS, byte-identical)

**From:** Team Blue
**To:** Team Red
**Re:** RED-RESP-STEP1-COLLAPSE-CANONICAL-PLACEMENT-AND-GATE-SEMANTICS (Option-1 ruling)
**Status:** W1.6 gate **PASS** — requesting Red confirmation of closure.

## Result

The pinned-path Option-1 gate ran on LUMI (job **18958015**, standard partition,
128 cores, elapsed 24:59). The pure 5-sigma z_comb verifier returned:

```
worst z_combined : 0.00
kicks > 3 sigma  : 0
kicks > 5 sigma  : 0
VERDICT          : PASS
```

`worst_z = 0.00` across all 60 kicks is byte-identity to the 4-decimal
aggregation precision, not merely statistical consistency. The candidate is the
sweep engine's own fresh output
(`results/gate_option1_18958015/byo_dat/floquet_dtc/QB11-...-QB26/device_calibrated/aggregated_autocorr.dat`,
80 simulations / 2400 tasks / 1444 s of real compute), compared against the
banked reference `examples/reference/floquet_dtc_q10_autocorr.csv` — a distinct
file, so this is reproduction, not self-comparison.

## What it establishes

Both arms were pinned to the canonical placement
`[QB11, QB5, QB6, QB7, QB13, QB21, QB29, QB28, QB27, QB26]`, both read the
banked file disorder, both built per-qubit T1/T2 from cal `08c3c70f`, and both
share the seed anchor `resolve_instance_seed(0, s)` (instance-0 = 3757552657).
The standalone runner (`floquet_runner_v2.py`, which banked the reference) and
the production sweep engine therefore drive the **same** `prepare_simulation`
seam, and the gate demonstrates that empirically: identical circuit (the patch-
18b builder-equivalence canary holds), identical noise model, identical seed
=> deterministic Aer => identical counts => `z = 0`. This is exactly the
reproduction Red's Option-1 collapse was defined to test.

## Conformance to the Option-1 ruling

- **Pure z_comb, 5-sigma** — gate mode only; `--mode step1-residual`, `--floor`,
  and `--max-rel-dev` are not passed (the old 2% ceiling is gone).
- **Both arms pinned to ONE canonical placement** — `physical_qubits` in
  `examples/byo/floquet_dtc_q10_sweep.yaml`; tied to the recorded `_CANONICAL`
  authority by `tests/unit/test_canonical_placement_guard.py`.
- **master_seed pre-flight** — `tests/_w1_gate_preflight.py` asserts the file
  `_meta.master_seed` resolves to int 0 before launch (guards the silent-entropy
  desync in the worker's seed fallback).
- **Reference regenerated + superseded** — file-disorder + pinned canonical,
  banked at commit a7d8c50 (jobs 18957439 device-cal / 18957440 noiseless /
  18957722 aggregate-build), superseding the legacy-draw / free-layout cb33530
  reference, with a provenance sidecar.

## Residual hazard (not blocking)

The two entry points remain distinct code paths (the **D9** hazard). This gate
pins their agreement empirically rather than retiring a path, so D9 stays open
and tracked. Drift is guarded structurally by the builder-equivalence canary
(`test_runner_and_byo_factory_build_identical_circuits`) and the canonical
placement guard; both run with the unit suite.

## Request

Blue considers W1.6 closed and asks Red to confirm, so the team can move to the
D7 multi-observable echo engine (the researcher's AK10 autocorrelator-vs-echo
campaign) and, after that, solver-driven placement selection.
