# FINDING — PadDelay silently skips Delay insertion under the device-calibrated path; idle-time decoherence does not fire in production sweeps

Status: NEW (verified, fix proposed; production change pending review)
Date:   2026-05-27
Branch: feature/device-calibrated-noise (HEAD at time of finding: `fe22fe5`)
Owner:  Michael Mucciardi
Scope:  All device-calibrated runs through `prepare_simulation`, including every BYO/Floquet sweep landed on this branch.

---

## 1. Executive summary

The device-calibrated path's `RelaxationNoisePass` (`op_types=[Delay]`, configured by `build_relaxation_pass` in `src/lumi_hpc_qc/backends/device_noise.py`) is documented as the mechanism that decoheres the **variable-duration idle steps the ALAP scheduler inserts** between gates. It never actually fires in production. The chain is correct end-to-end except for one omission: `_NATIVE_BASIS` in `src/lumi_hpc_qc/backends/prepare.py:50` does not include `"delay"`, so the `Target` built by `Target.from_configuration(basis_gates=_NATIVE_BASIS, …)` reports `delay` as unsupported, and PadDelay (`qiskit.transpiler.passes.scheduling.padding.base_padding.BasePadding._pad`) silently skips Delay insertion on every qubit whose Target says delay isn't supported. The scheduled circuits that reach Aer therefore contain no idle Delay instructions, and `RelaxationNoisePass(op_types=[Delay])` has nothing to decohere.

Net effect: device-calibrated sweeps include the resident gate-time thermal relaxation on `r/sx/x/cz/measure` (which is keyed to gate ops, not delays, and is unaffected), but the variable idle-time relaxation component is **silently dropped**. Decoherence is under-represented relative to what the codebase advertises and intends.

Surfaced by the §2.1 integration test `test_idle_relaxation_tracks_placement_through_full_schedule`, which traverses the production scheduler path (unlike the existing tests, which either bypass PadDelay or don't measure decoherence magnitudes). All three attempts at the worker time-filler (`id`, `sx`, `sx`+`barrier(1)`) produced identical target survivals (~0.87 with readout), back-solving to ~140 ns of |1⟩ exposure — i.e., X + CZ + CZ gate-time only, **with the 5 µs of worker-occupied time contributing zero idle decoherence on the target**.

---

## 2. Root cause — code-path walk

All file/line references verified against the pinned trees in this session:
- Qiskit 2.3.0 source (uploaded).
- qiskit-aer (uploaded; `RelaxationNoisePass` body at `qiskit_aer/noise/passes/relaxation_noise_pass.py`).
- HPCQC at `fe22fe5`.

**Step 1 — Target built without `delay`.**
`src/lumi_hpc_qc/backends/prepare.py:50`
```python
_NATIVE_BASIS = ["r", "rz", "sx", "x", "cz", "id", "measure"]
```
`src/lumi_hpc_qc/backends/prepare.py:319-325`
```python
target = Target.from_configuration(
    basis_gates=_NATIVE_BASIS,
    num_qubits=num_qubits,
    coupling_map=cmap,
    instruction_durations=instr_durations,
    dt=dt_s,
)
```
`qiskit/transpiler/target.py:from_configuration` adds only the gates listed in `basis_gates` (via `name_mapping = get_standard_gate_name_mapping()`). Delay is in that mapping (`Delay(time)`), but it is **not added unless it appears in `basis_gates`**. So the resulting Target's `operation_names` set does not contain `"delay"`.

**Step 2 — `instr_durations` lists "delay" but that does not register the instruction.**
`src/lumi_hpc_qc/backends/prepare.py:300-309` builds an `InstructionDurations` table that includes `("id", None, sg)` but no `"delay"` entry. Even if `"delay"` *were* in that table, `InstructionDurations` is consulted only to look up durations for instructions that are *already* in the Target. The set of "supported instructions on the Target" comes from `basis_gates`, not from `instr_durations`.

**Step 3 — opt-level-0 passes preserve the input.**
`qiskit/transpiler/preset_passmanagers/builtin_plugins.py`:
- `DefaultInitPassManager` at level 0 runs only `common.generate_unroll_3q` (no 1q-fusion, no identity-removal).
- `OptimizationPassManager` at level 0 is **empty** (`if optimization_level != 0: …` — the entire optimization block is skipped).
- `RemoveIdentityEquivalent` runs only at levels 2 and 3; `Optimize1qGatesDecomposition` only at 1/2/3.

So the worker chains in the three integration-test attempts (`id`, `sx`, `sx+barrier(1)`) were **all preserved** through translation. The collapse we hypothesized in the previous iterations was wrong; this finding supersedes that diagnosis.

**Step 4 — ALAP computes the schedule correctly.**
`qiskit/transpiler/passes/scheduling/scheduling/alap.py:ALAPScheduleAnalysis.run` populates `property_set["node_start_time"]` with the correct schedule. The worker is busy for `N * single_gate_time_ns` between the two CZ anchors; the target's idle gap on its wire equals that duration.

**Step 5 — PadDelay silently refuses to insert Delays.** The smoking gun.
`qiskit/transpiler/passes/scheduling/padding/base_padding.py` (`BasePadding`, parent of `PadDelay`):
```python
def __delay_supported(self, qarg: int) -> bool:
    """Delay operation is supported on the qubit (qarg) or not."""
    if self.target is None or self.target.instruction_supported("delay", qargs=(qarg,)):
        return True
    return False
```
And in the same file's `run` body:
```python
if t0 - idle_after[bit] > 0 and self.__delay_supported(dag.find_bit(bit).index):
    # ... call self._pad(...) which inserts a Delay
```
Since the device-calibrated Target's `operation_names` lacks `"delay"`, `target.instruction_supported("delay", qargs=(qarg,))` returns False for every qubit, `__delay_supported` returns False, and **the `_pad` call is never made**. The schedule is otherwise correct; the Delay-insertion is just skipped.

**Step 6 — RelaxationNoisePass has nothing to act on.**
`qiskit-aer/qiskit_aer/noise/passes/relaxation_noise_pass.py`:
```python
super().__init__(self._thermal_relaxation_error, op_types=op_types, method="append")
```
And `_thermal_relaxation_error` is only invoked for ops whose type is in `op_types`. In our config (`device_noise.build_relaxation_pass`), `op_types=[Delay]`. With no Delay nodes in the scheduled DAG, the pass is a no-op.

**Step 7 — Resident gate-time relaxation still fires (this is what the integration test measured).**
`device_noise.build_control_readout_noise_model` attaches gate-time thermal relaxation to `_TIMED_1Q_GATES = ["r","sx","x"]` and `_NATIVE_2Q_GATES = ["cz"]`. So the target's X + 2×CZ contributed ~140 ns of |1⟩ exposure regardless of placement, exactly matching the observed `p0_lossy ≈ 0.870` across all three attempts (pre-readout 0.92 → `t = -T1·ln(0.92)/T1=...` consistent with ~170 ns of exposure once measurement-prep alignment is included).

---

## 3. Empirical evidence — three sbatch runs

Jobs `18873650` (id), `18873763` (sx, opt 0), `18873828` (sx + `barrier(1)`). All three ran on LUMI in container `/appl/local/quantum/qiskit/qiskit_2.3.0_csc.sif`, 4096 shots, seed 1234, on `feature/device-calibrated-noise`.

|  Worker chain                | `p0_lossy` (idx0 = QB35) | Implied target |1⟩ exposure |
|------------------------------|--------------------------|----------------------------|
| 250 × `id`                   | 0.8682                   | ~175 ns                    |
| 250 × `sx`                   | 0.8699                   | ~170 ns                    |
| 250 × (`sx` + `barrier(1)`)  | 0.8699                   | ~170 ns                    |

Same magnitude to within shot noise (~0.005). The worker chain duration (5 µs in each case) contributed **zero** to the target's exposure. Consistent only with "no Delay on the target wire."

The other 24 tests (gate test, magnitude, layout, threading, master_seed, BYO storage) passed in every run.

---

## 4. Production impact

Every device-calibrated sweep landed on this branch (Phase E / D3.x and earlier) ran without idle-time decoherence. The values for `aggregated_autocorr.dat` and the `examples/reference/floquet_dtc_q10_*` baselines were produced under that broken regime. Specifically:

- Resident gate-time thermal relaxation on `r/sx/x/cz` — **fires correctly**, unaffected.
- Depolarizing on 1q/2q gates — **fires correctly**, unaffected.
- Readout error — **fires correctly**, unaffected.
- Variable idle-time relaxation (RelaxationNoisePass on scheduled delays) — **silently does not fire**.

For dense DTC circuits the missing component is small per-kick (idle gaps are tens to hundreds of ns; resident gate-time relaxation is the dominant noise source), but it compounds over 60 kicks, and the missing component is precisely the one that distinguishes lossy from ideal qubits at the *idle-window* scale. Any conclusion about placement sensitivity drawn from current device-calibrated benchmarks should be revisited after the fix.

---

## 5. Proposed fix

One line change in `src/lumi_hpc_qc/backends/prepare.py:50`:

```diff
-_NATIVE_BASIS = ["r", "rz", "sx", "x", "cz", "id", "measure"]
+_NATIVE_BASIS = ["r", "rz", "sx", "x", "cz", "id", "measure", "delay"]
```

`"delay"` is in `qiskit.circuit.library.standard_gates.get_standard_gate_name_mapping` (returns `Delay(time)`), so `Target.from_configuration` accepts it as a 1-qubit variable-width instruction with `duration=None` (variable). With "delay" in the Target:
- `PadDelay.__delay_supported(qarg)` returns True for every qubit.
- PadDelay inserts Delay instructions into the scheduled circuit's idle gaps with their computed durations in dt.
- `RelaxationNoisePass(op_types=[Delay], dt=1e-9, t1s, t2s)` acts on each Delay, applying `thermal_relaxation_error(t1, t2, duration_seconds)` keyed to the qubit index — i.e., `physical_qubits[k]` after the F5a placement.
- The integration test (now or after revert + reintroduction) would pass against bands ~0.13 / ~0.85.
- Production sweeps would correctly model the variable idle-time decoherence the codebase already advertises.

Note: this also makes the device-calibrated path accept explicit `Delay` instructions in input circuits (they no longer get rejected by `BasisTranslator`). The explicit-Delay tests would still pass and could in principle be hardened to go through the full `prepare_simulation` pipeline rather than running on `prep.simulator` directly — though that hardening is optional, the §2.1 keying claim is already closed.

---

## 6. Required follow-up before merging the fix

This is not a drop-in fix because it **changes the noise model**. Before merging:

1. **Re-baseline reference data.** Re-generate `examples/reference/floquet_dtc_q10_autocorr.csv` and any `phys_device-calibrated_*` aggregated outputs on the fixed branch. Old reference values were produced under the broken regime and will not match the corrected ones.

2. **F4 byte-identicality.** The F4 noiseless baseline (the `physical_qubits=None` free-layout path) is unaffected — F4 doesn't go through device-calibrated prepare. Confirm with one sbatch comparison.

3. **Gate-2 bit-exact reproduction.** The D3.5 gate-2 plan (q10, 60 kicks, 40 instances, shots-explicit, ±3·sem) needs to be re-anchored to fixed-regime reference data. The shape of the result (alternating-sign DTC autocorrelator) is unchanged; the numerical values will shift.

4. **Magnitude expectation.** The variable idle in DTC circuits is small per-kick (tens of ns) but per-kick gate-count and idle structure determine the cumulative shift over 60 kicks. Expect modest but non-zero changes in autocorrelator magnitudes, more pronounced when placements include lossy qubits like QB35.

5. **§2.1 integration test.** Re-enable `test_idle_relaxation_tracks_placement_through_full_schedule` (currently committed at `fe22fe5`); it should pass against the documented bands once the fix lands.

---

## 7. Validation plan once the fix lands

- Unit: `tests/unit/test_f5a_placement_noise.py` — expect 16 passed (including the integration test that currently fails). `tests/unit/test_d34c_byo_storage.py` — expect 9 passed (unaffected).
- Smoke: one device-calibrated BYO sweep at q10 with the same configs as the v1.1.0 release, compare autocorrelator magnitudes against the prior broken-regime baselines to quantify the shift.
- New regression: a minimal integration test that asserts `prep.run_circuits[0]` contains at least one `Delay` op after `prepare_simulation` runs with a multi-gate circuit (so this defect cannot silently regress in the future).

---

## 8. References

- `src/lumi_hpc_qc/backends/prepare.py:50, :319-325, :386-410` (HPCQC, `fe22fe5`).
- `src/lumi_hpc_qc/backends/device_noise.py:339-403` (HPCQC, `fe22fe5`) — `build_relaxation_pass` with `op_types=[Delay]`.
- `qiskit/transpiler/passes/scheduling/padding/base_padding.py` — `__delay_supported`, `_pad` gating.
- `qiskit/transpiler/passes/scheduling/padding/pad_delay.py` — `PadDelay._pad` body.
- `qiskit/transpiler/preset_passmanagers/builtin_plugins.py` — opt-level-0 init + optimization stages (no 1q-fusion).
- `qiskit/transpiler/preset_passmanagers/common.py:651` — `generate_scheduling` (alap → TimeUnitConversion + ALAPScheduleAnalysis + PadDelay).
- `qiskit/transpiler/target.py:792` — `Target.from_configuration` (no auto-Delay).
- `qiskit_aer/noise/passes/relaxation_noise_pass.py` — `RelaxationNoisePass`, `op_types` gating.
- Slurm jobs: `18873650`, `18873763`, `18873828`.
