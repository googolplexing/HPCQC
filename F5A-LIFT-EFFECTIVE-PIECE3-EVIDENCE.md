# F5a lift effective — Piece 3 diff + evidence

Review record for the F5a single-placement guardrail lift. The lift was authorized
by `F5A-LIFT-APPROVED` and takes effect on review of this diff; the requirements
below are that decision's §4 conditions, restated as the bar the diff must clear.
Evidence detail and reproduction: `evidence/F5A-PIECE2-PIECE3-PROVENANCE.md`.

## What the diff does
Lifts the device-calibrated single-placement clamp in the BYO executor
(`sweep_engine.py`): removes `max_placements = 1 if wants_device_cal` and the hard
error on >1 manual placement under device_calibrated. Wires the placement union
(`solver_top_n` from config when both a manual set and a solver top-N are supplied).
Replaces the guardrail-keyed `noise_placement_independent` with a pure helper
`_noise_placement_independent(env_source, len(placements))`: true iff
device_calibrated AND exactly one resolved placement.

## Evidence (cal 08c3c70f, autocorr, 10 seeds × 40 kicks × 1000 shots)
1. **Flag truth table** — unit test 4/4 (`tests/unit/test_f5a_lift_flag.py`):
   device_calibrated@1 → True; @{2,5,30} → False; noiseless@any → False.
2. **Single-placement byte-identical** — lifted single HIGH run (job 18986846)
   `byo_dat` diffs empty vs banked pre-lift run (18984820); HDF5 flag
   device_calibrated → True (10). The 1-placement path is unchanged in data and flag.
3. **Multi-placement runs with the clamp lifted; per-placement composition**
   (job 18986847, `examples/byo/floquet_dtc_q10_f5a_multiplacement.yaml` — HIGH+LOW
   as two manual placements in one device_calibrated sweep, the config the clamp used
   to reject): runs cleanly (`2 manual placement(s)`, 40 units, 0 errors); HIGH and
   LOW subtrees byte-identical to the isolated single runs (18984820 / 18984821) —
   per-placement composition with zero cross-talk, proven not asserted; HDF5 flag
   device_calibrated → False (20).

## Requirements map (from F5A-LIFT-APPROVED §4)
- **Relax clamp scoped to per-placement composition** — done. The BYO device-cal path
  composes per placement via `build_control_readout_noise_model(physical_qubits=chain)`
  as its only mechanism; check 3's byte-identity to the isolated runs proves it.
- **noise_placement_independent false on multi-placement device-cal records** — done;
  measured False on all 20 (check 3).
- **Preserve single-placement byte-identical, test-assert** — done; check 2 (empty
  diff + flag True) and the unit test pin it.
- **Do NOT remove/weaken per-placement composition** — untouched; it is the mechanism
  the lift relies on (byte-identity confirms it still runs per placement).
- **Do NOT lift unvalidated conditions** — scoped in code to the per-placement-composed
  device_calibrated path only; shared-model and QPU multi-placement are explicitly
  excluded (separate review).
- **Do NOT silently change the single-placement path** — byte-identical, flag still True.

## Caveats carried forward (neither blocks the lift)
- Solver top-N returns N variations of one device locale, not N regions: on Q50 the
  top-4 max_fidelity chains share 8/10 qubits (one well-calibrated corner). Dedup is
  correct (set-level); a researcher wanting distinct placements should know top-N
  clusters. A diversity/disjointness option is a possible follow-up.
- Byte-identity is currently a hand-run diff; recommend promoting check 3 to a CI
  integration test so the no-cross-talk guarantee is enforced rather than re-verified
  by hand.

## Disposition
Single-placement path byte-for-byte unchanged (data + flag); multi-placement path runs
with each placement composed from its own qubits, proven by byte-identity to the runs
done in isolation, records honestly flagged placement-dependent. On sign-off the lift
is effective and the multi-placement device-cal campaign (manual ∪ solver-top-X chains)
is cleared to run.
