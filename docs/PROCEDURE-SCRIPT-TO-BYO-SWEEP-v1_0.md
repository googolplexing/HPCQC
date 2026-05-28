# PROCEDURE — Converting a research script to an HPCQC BYO sweep (with disorder)

**Version:** 1.0
**Date:** 2026-05-28
**Base HEAD:** `f0ce463`
**Worked example:** `Floquet_DTC_AK10_echo.py` (echo variant; 10q, 40 kicks,
100 shots, 10 instances, polarized init, `default_rng(1234)` disorder)
**Companion design:** `DESIGN-MULTI-OBSERVABLE-BYO-ECHO-v1_0.md`
**Companion walkthrough:** `docs/MIGRATION_FloquetDTC.md` (AK7 single-observable)

This document answers a researcher's practical question: *"I have a standalone
script like `Floquet_DTC_AK10_echo.py`. How do I turn it into the three things
HPCQC needs — a circuit factory, a sweep YAML, and a disorder JSON — and how do
I generate that disorder JSON without hand-typing hundreds of angles?"*

It is written so a researcher can follow it WITHOUT assistance, given the repo
at `f0ce463`. It folds in three findings about the disorder `generate` path
that change what is and isn't possible today (§4).

---

## 0. The mental model: three files, three jobs

A standalone research script bundles four things that HPCQC wants separated:

| In the script | Becomes | Why separated |
|---|---|---|
| the circuit-building gates | **circuit factory** (`.py`) | pure function of data; no RNG, no I/O |
| the swept axis (e.g. `num_kicks` loop) | **`grid:` in the YAML** | the engine enumerates it |
| fixed knobs (`epsilon`, `num_qubits`) | **`fixed:` in the YAML** | constant across the sweep |
| the per-instance disorder draw (`rng.uniform(...)`) | **disorder JSON** (`source: file`) | frozen, reproducible, auditable |

The factory says *what the circuit is*. The YAML says *what you sweep and under
what noise*. The disorder JSON says *what random realizations you used*. The
engine combines all three: for each (seed, grid point, noise env) it calls
`factory(**fixed ∪ disorder_instance ∪ grid_point)`, simulates, aggregates.

---

## 1. Step A — Extract the circuit factory

Take the script's circuit-building code and make it a **pure, keyword-only
function** that receives every parameter as data and draws no randomness.

For AK10, the circuit body is `apply_one_floquet_period` +
`build_autocorr_circuit` (and `build_echo_circuit` for the echo variant). The
factory inlines those with the disorder arriving as arguments:

```python
# examples/byo/floquet_dtc_echo.py  (already built + verified this session)
def build_circuit(*, num_kicks, epsilon, num_qubits,
                  hz_angles, Jzz_angles, init_bit_array) -> QuantumCircuit:
    h_x = (1.0 - epsilon) * np.pi
    qc = QuantumCircuit(num_qubits, num_qubits)
    for w in range(num_qubits):
        if init_bit_array[w] == 1:
            qc.x(w)
    for _ in range(num_kicks):
        for w in range(num_qubits):            qc.rx(h_x, w)
        for w in range(num_qubits):            qc.rz(hz_angles[w], w)
        for w in range(num_qubits - 1):        qc.rzz(Jzz_angles[w], w, w + 1)
    qc.measure(range(num_qubits), range(num_qubits))
    return qc
```

Rules the factory must obey (enforced by `circuit_loader` + the §7.5.1 signature
check):
- **Keyword-only** (`*,`) — the engine spreads `**params`.
- **Returns exactly one `QuantumCircuit`** (single-observable contract). The
  echo variant exposes a SECOND function `build_circuit_echo`; the engine's
  multi-observable consumption of two functions is design-staged work (see
  §5 and the design doc), so for a single-observable sweep today you point at
  ONE function.
- **Import-safe** — no work at module scope (the loader executes the module on
  import). AK10's `if __name__ == "__main__": main()` guard already ensures
  this; keep it.
- **Logical gates only** (`rx`/`rz`/`rzz`) — do NOT pre-transpile to the device
  native basis. Native lowering happens per-arm downstream (device_calibrated
  → prx/cz with calibrated durations; noiseless → statevector).

The argument NAMES matter: `hz_angles`, `Jzz_angles`, `init_bit_array` must
match the keys in the disorder JSON (§3) exactly — the signature check fails the
sweep at config time if they don't.

---

## 2. Step B — Write the sweep YAML

The YAML partitions the script's parameters into `grid` / `fixed` / `disorder`
and declares the noise arms. For AK10:

```yaml
sweep:
  experiments:
    - type: byo_circuit
      label: floquet_dtc_echo_ak10
      circuit_script: examples/byo/floquet_dtc_echo.py
      circuit_function: build_circuit          # single observable (autocorr)
      fixed:
        num_qubits: 10                          # AK10 num_qubits
        epsilon: 0.03                           # AK10 epsilon
      grid:
        num_kicks: {range: [0, 40]}             # AK10 num_max_kicks=40 -> 0..39
      disorder:
        source: file                            # see §4: 'file' is the only
        file: examples/byo/floquet_disorder_q10_echo_ak10.json   # working path today
        initial_state: 3                        # AK10 Initial_state=3 (polarized)
      disorder_gates: [rz, rzz]                 # which gates carry disorder angles
      seed_list: [0,1,2,3,4,5,6,7,8,9]          # AK10 num_gate_instances=10
      noise_configs: [noiseless, device_calibrated]
  calibrations:
    - examples/q50_calibration_20260524_08c3c70f.json
```

Mapping back to AK10's top-of-file constants:
- `num_qubits=10` → `fixed.num_qubits`
- `epsilon=0.03` → `fixed.epsilon`
- `num_max_kicks=40` → `grid.num_kicks.range: [0, 40]` (stop-EXCLUSIVE → 0..39)
- `num_gate_instances=10` → `seed_list` of length 10
- `Initial_state=3` → `disorder.initial_state: 3`
- `backend_code=1` (AerSimulator, noiseless) → the `noiseless` arm
- `num_shots=100` → see §4 finding #2: **the `shots` field is not yet read on
  the BYO path; it currently runs at the hardcoded `byo_shots=1000`.**

---

## 3. Step C — Generate the disorder JSON (NOT by hand)

The disorder JSON is a frozen record of the per-instance random draws. Schema:

```json
{
  "_meta": { "master_seed": 1234, "num_qubits": 10, "initial_state": 3, ... },
  "instances": {
    "0": {"hz_angles": [...10...], "Jzz_angles": [...10...], "init_bit_array": [...10...]},
    "1": {...},
    ...
    "9": {...}
  }
}
```

A researcher does **NOT** type this. They generate it with a ~15-line script
that reuses their ORIGINAL draw code, dumping to JSON instead of feeding a
circuit. The key discipline: **reproduce the original RNG mechanism and call
order EXACTLY**, or the numbers won't match the standalone script.

For AK10, the original draw (in `main()`) is a single `default_rng(1234)` stream
advanced serially, Jz before hz, per instance, polarized init drawing nothing:

```python
# gen_disorder_ak10.py  — run once, locally, commit the OUTPUT json
import numpy as np, json

NUM_QUBITS, NUM_INSTANCES, SEED = 10, 10, 1234
rng = np.random.default_rng(SEED)            # ONE stream, exactly as AK10 main()
instances = {}
for n in range(NUM_INSTANCES):
    # ORDER MATTERS — AK10 draws Jz THEN hz each instance:
    Jz = rng.uniform(-1.5*np.pi, -0.5*np.pi, NUM_QUBITS)   # AK10 'Jz_angles'
    hz = rng.uniform(-np.pi,      np.pi,      NUM_QUBITS)   # AK10 'hz_angles'
    instances[str(n)] = {
        "hz_angles":  [float(x) for x in hz],
        "Jzz_angles": [float(x) for x in Jz],   # AK10's Jz_angles ARE the rzz couplings
        "init_bit_array": [0]*NUM_QUBITS,        # Initial_state=3 polarized; no draw
    }
json.dump({"_meta": {"master_seed": SEED, "num_qubits": NUM_QUBITS,
                     "initial_state": 3,
                     "rng": "numpy.random.default_rng(1234), single serial stream",
                     "draw_order": "Jz=uniform(-1.5pi,-0.5pi,10) THEN hz=uniform(-pi,pi,10)"},
           "instances": instances},
          open("examples/byo/floquet_disorder_q10_echo_ak10.json", "w"), indent=2)
```

Three subtleties that bite if missed (all AK10-specific, generalize by reading
your own script's `main()`):
1. **Variable-name trap.** AK10's variable is `Jz_angles` but it is passed as
   the `rzz` two-qubit coupling, i.e. the factory's `Jzz_angles`. The JSON key
   is `Jzz_angles`. Get this wrong and the couplings/fields swap silently.
2. **Draw order.** AK10 draws Jz BEFORE hz each instance. The single stream
   means instance N depends on all prior draws; wrong order → wrong numbers
   from instance 0 onward.
3. **Init draws nothing.** `Initial_state=3` (polarized) is deterministic
   all-zeros; it does NOT touch the rng. (`Initial_state=1`, random, WOULD draw
   from Python's `random` — a separate stream — and you'd have to reproduce that
   too. AK10's echo run uses 3.)

PCG64 output is numpy-version-stable (numpy guarantees bit-generator
reproducibility across versions for a fixed seed), and because you commit the
OUTPUT JSON, faithfulness is frozen into the artifact regardless of the numpy on
the run machine. Record the generating numpy version in `_meta` for provenance.

This `gen_disorder_ak10.py` is a throwaway local generator — it does NOT go in
the repo as a code path; only its JSON output is committed. (Contrast with a
`sample_disorder` factory hook, §4.)

---

## 4. THREE FINDINGS about the disorder `generate` path (read before relying on it)

A researcher might reasonably hope to skip the JSON entirely via
`disorder: {source: generate, ...}`. Here is the true state at `f0ce463`:

**Finding #1 — `source: generate` is NOT wired into the engine today.**
`resolve_disorder` (in `byo_sweep.py`) supports a `generate` branch, but it
requires a `sampler(rng, num_qubits) -> dict` callable, and the engine's call
site (`sweep_engine.py:430`) invokes `resolve_disorder(...)` WITHOUT passing a
`sampler`. The `generate` branch raises `ValueError` if `sampler is None`. So
end-to-end through the engine, **`source: file` is the only functional disorder
path.** The disorder JSON is therefore MANDATORY for any BYO sweep right now,
not merely an AK10-replication convenience.

**Finding #2 — the BYO `shots` field is inert.** `sweep_engine.py:2061`
hardcodes `byo_shots = 1000` as the fallback for envs whose own `shots == 0`
(the noiseless arm). The experiment-level `shots:` key is not read by the BYO
parser (lines ~631–640 read `circuit_function`/`disorder_gates`/`seed_list` but
not `shots`). So a YAML asking for `shots: 100` silently runs at 1000. AK10
shot-count fidelity requires wiring this field — design increment 0.

**Finding #3 — even when wired, `generate` cannot reproduce AK10's RNG.** The
`_spawn_rng` helper offers two generators: `pcg64` does
`SeedSequence(master_seed).spawn(seed_index+1)[seed_index]` — a per-seed
INDEPENDENT child stream; `legacy_npr` reseeds the global `np.random` per
instance from a SeedSequence-derived uint32. AK10 uses a SINGLE
`default_rng(1234)` stream advanced serially across all instances. Neither
generator reproduces that single-stream semantics, so neither yields AK10's
exact angles. **A new generator option (e.g. `generator: shared_default_rng`)
must be added to `_spawn_rng`** to replicate the single-stream `default_rng`
semantics if `generate` is ever to match AK10. Until both Finding #1 (wire the
sampler) and Finding #3 (add the shared-stream generator) are addressed, the
frozen JSON (§3) is the only way to get AK10's exact disorder.

### What this means for the researcher's two scenarios

- **"Reproduce AK10's exact numbers" (standalone fidelity study):** must use the
  frozen JSON from §3. `generate` cannot do it today (Findings #1 + #3). This is
  the path we're on.
- **"My own study, any valid disorder, don't care about matching a prior script":**
  would naturally use `generate` with a `sample_disorder` hook — but that hook
  is not wired (Finding #1). So even this researcher must, today, either (a)
  generate a frozen JSON with their own ~15-line script (the §3 pattern with
  their own ranges/seed), or (b) wait for the `generate` wiring. Option (a) is
  the pragmatic answer and is what the existing committed configs
  (`floquet_disorder_q4.json`, `floquet_disorder_q10.json`) all do.

**Bottom line for the researcher:** yes, you generate a disorder JSON — but with
a tiny throwaway script that reuses your draw code, never by hand. And it is
currently REQUIRED (not optional), because `source: generate` is not engine-wired
at `f0ce463`.

---

## 5. Step D — Run it on LUMI

Once factory + YAML + disorder JSON are committed and pulled to LUMI:

```bash
cd /flash/project_462001289/mucciard/HPCQC/HPCQC
# quick q4-scale smoke first if you made one; else the real config:
sbatch --account=project_462001289 --partition=standard \
  --nodes=1 --ntasks-per-node=1 --cpus-per-task=128 \
  --time=06:00:00 --job-name=floquet_echo_ak10 \
  --wrap='cd "$SLURM_SUBMIT_DIR" && source "$SLURM_SUBMIT_DIR/env.sh" && \
    export SINGULARITYENV_PYTHONPATH="$HPCQC_ROOT/src" && \
    srun $HPCQC_CPU_WRAPPER $HPCQC_CPU_CONTAINER bash -c \
      "python3 -m lumi_hpc_qc.sweep.run_sweep examples/byo/floquet_dtc_q10_echo_sweep.yaml"'
```

Before re-running on a tree with prior output, move it aside (the resume cache
short-circuits on completed task IDs in `sweep_output/sweep.h5`):

```bash
mv sweep_output sweep_output.$(date +%s)-old
```

Budget: the device_calibrated arm is ~48× slower per instance than noiseless
(post-PadDelay-fix Kraus pipeline). For AK10's noiseless-only fidelity check, it
is fast; the device_calibrated arm is the added cost.

---

## 6. The echo (two-observable) wrinkle — why this is more than a file conversion

AK10 is not a single-observable script: it computes the autocorrelator AND an
echo, then plots their ratio `A(0)A(T)/A_0`. That needs TWO circuits per (seed,
kick) plus a derived ratio — and the current BYO engine path is
single-circuit-per-task. The factory already exposes both functions
(`build_circuit`, `build_circuit_echo`); the engine work to consume two
observables and derive the ratio is staged in
`DESIGN-MULTI-OBSERVABLE-BYO-ECHO-v1_0.md` (Option A; increments 0,3,4,5).

Until that lands, a researcher has two faithful options:
- **Single-observable now:** point `circuit_function` at `build_circuit` and run
  just the autocorrelator (the period-doubling signal). No echo normalization.
- **Option B fallback (both observables, offline ratio):** declare TWO
  single-observable experiments in one YAML (one per function, same seeds /
  disorder / grid / shots / calibration), run both on today's engine, and
  compute the ratio offline keyed on (seed, num_kicks, placement, env). The
  echo sweep YAML carries this fallback as a commented block.

---

## 7. Checklist (researcher, unaided, at f0ce463)

1. [ ] Extract circuit body → pure keyword-only `build_circuit` in
       `examples/byo/<name>.py`. Logical gates only. Import-safe.
2. [ ] Confirm factory arg names == disorder JSON keys (`hz_angles`,
       `Jzz_angles`, `init_bit_array`).
3. [ ] Write a throwaway `gen_disorder_<name>.py` that reuses your script's
       EXACT draw code/order/seed; dump to
       `examples/byo/floquet_disorder_<name>.json`. Commit the JSON, not the
       generator.
4. [ ] Write `examples/byo/<name>_sweep.yaml`: `fixed` / `grid` / `disorder
       (source: file)` / `seed_list` / `noise_configs` / `calibrations`.
5. [ ] Note Finding #2: `shots:` is inert; runs at 1000 until wired.
6. [ ] Commit on Mac, `git pull --ff-only` on LUMI.
7. [ ] LUMI smoke at small scale; then the full sbatch (§5).
8. [ ] If you need the echo ratio: use Option B fallback or wait for the
       multi-observable increments (§6).
