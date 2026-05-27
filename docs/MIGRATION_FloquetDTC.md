# Migrating a research script to an HPCQC BYO sweep — Floquet DTC

**What this shows:** how to take a standalone Qiskit research script
(`Floquet_DTC_AK7.py`, the discrete time-crystal driver) and split it into the
two artifacts HPCQC's parameterized BYO path consumes:

1. a **pure circuit-factory module** — `examples/byo/floquet_dtc.py`
2. a **sweep YAML** — `examples/byo/floquet_byo_sweep.yaml` (+ a per-seed
   disorder file `floquet_disorder_q4.json`)

HPCQC then handles the sweep, placement solving, device-calibrated noise, and
the counts→autocorrelator observable that the original script computed by hand.

> The autocorrelator HPCQC computes is the **same function** as the original's
> `get_autocorrelation` — byte-identical (`floquet_runner_v2.py:94`). You are not
> re-deriving the physics; you are handing the build + sweep + noise + averaging
> to the framework.

---

## The core idea: separate *what the circuit is* from *what you sweep*

The original script mixes four concerns in one file: circuit construction, the
parameter sweep (`num_kicks`), per-instance disorder (RNG), and the
observable/averaging. HPCQC wants these separated:

| Concern in the original | Where it goes in HPCQC |
|---|---|
| `build_circuit(...)` / `apply_one_floquet_period(...)` | the **factory** `build_circuit` (kwargs only) |
| `for n_kicks in range(num_max_kicks)` | YAML `grid: num_kicks` |
| `num_qubits`, `epsilon` constants | YAML `fixed:` |
| `np.random.uniform(...)` per instance | YAML `disorder:` (per-seed file/generate) |
| `for n in range(num_gate_instances)` | YAML `seed_list:` (averaged) |
| `get_autocorrelation(...)` + `/ num_gate_instances` | the engine (same formula) |
| `AerSimulator` / `IQMFakeAphrodite` | YAML `noise_configs:` |

## Step 1 — make the build function pure and keyword-only

This is the one refactor that matters. In the original, the build path is pure
*per instance* but the **randomness lives in the main loop**, not the build:

```python
# Floquet_DTC_AK7.py — disorder drawn in the instance loop, OUTSIDE build_circuit
for n in range(num_gate_instances):
    Jz_angles = np.random.uniform(-1.5*np.pi, -0.5*np.pi, num_qubits)
    hz_angles = np.random.uniform(-np.pi, np.pi, num_qubits)
    circuits = [build_circuit(n_kicks, hz_angles, Jz_angles, init_bit_array)
                for n_kicks in range(num_max_kicks)]
```

Notice the structure the original *already* has, which is exactly the HPCQC
contract: **one disorder draw is shared across all `num_kicks` circuits in an
instance, and a fresh draw is taken per instance.** HPCQC formalizes this as
"disorder shared by identity across the grid, fresh per seed." The migration is
just moving the draw *out* of the script's loop and *into* per-seed data, so the
factory itself draws nothing:

```python
# examples/byo/floquet_dtc.py — pure, keyword-only; disorder arrives as data
def build_circuit(*, num_kicks, epsilon, num_qubits,
                  hz_angles, Jzz_angles, init_bit_array) -> QuantumCircuit:
    h_x = (1.0 - epsilon) * np.pi          # derived constant, computed here
    qc = QuantumCircuit(num_qubits, num_qubits)
    for w in range(num_qubits):
        if init_bit_array[w] == 1:
            qc.x(w)
    for _ in range(num_kicks):
        for w in range(num_qubits):
            qc.rx(h_x, w)                  # transverse kick (excluded from disorder sig)
        for w in range(num_qubits):
            qc.rz(hz_angles[w], w)         # hz disorder
        for w in range(num_qubits - 1):
            qc.rzz(Jzz_angles[w], w, w + 1)  # Jzz disorder
    qc.measure(range(num_qubits), range(num_qubits))
    return qc
```

The gate body is a line-for-line inline of the original's
`apply_one_floquet_period` + `build_circuit`. What changed:

- **`*` makes every parameter keyword-only** — HPCQC validates the signature
  against your YAML keys at submit time and rejects mismatches loudly.
- **No `np.random`, no module globals.** `num_qubits`/`epsilon` become kwargs;
  `h_x` is derived inside. If the factory drew RNG, HPCQC's cross-grid identity
  check would catch it: it builds two grid points and compares their
  disorder-bearing gates (`rz`/`rzz`); an impure factory yields different
  angles and the sweep **raises before running**. (The `rx` drive is excluded
  from that check, so promoting `epsilon` to a swept axis later is safe.)
- **`num_kicks=0` is allowed** — it builds an init-only circuit (no `rz`/`rzz`),
  the t=0 autocorrelator reference. The cross-grid check compares the two
  *highest* grid points precisely so this empty low point isn't flagged.

## Step 2 — move the disorder draw into a per-seed file

The original's two `np.random.uniform` calls become per-seed entries. Each
"gate instance" `n` is a seed; each seed carries the `hz_angles`/`Jzz_angles`/
`init_bit_array` that instance would have drawn:

```json
// examples/byo/floquet_disorder_q4.json
{
  "_meta": {"generator": "example", "master_seed": 0,
            "num_qubits": 4, "initial_state": 3},
  "instances": {
    "0": {"hz_angles": [0.860556, -1.446473, -2.884148, -3.037746],
          "Jzz_angles": [-2.157425, -1.844883, -2.806586, -2.420608],
          "init_bit_array": [0, 0, 0, 0]},
    "1": { ... },
    "2": { ... }
  }
}
```

`init_bit_array` here is the polarized state (`Initial_state == 3` → all zeros),
matching the original's `init_qubits_array`. You can either bake disorder into a
file like this (`source: file`, reproducible, what gate-2 uses) or have HPCQC
generate it deterministically per seed (`source: generate` with a `master_seed`).

## Step 3 — write the sweep YAML

```yaml
# examples/byo/floquet_byo_sweep.yaml
sweep:
  experiments:
    - type: byo_circuit
      label: floquet_dtc_example
      circuit_script: examples/byo/floquet_dtc.py
      circuit_function: build_circuit
      fixed:
        num_qubits: 4
        epsilon: 0.03
      grid:
        num_kicks: {range: [0, 6]}          # stop-exclusive -> 0..5
      disorder:
        source: file
        file: examples/byo/floquet_disorder_q4.json
        initial_state: 3
      disorder_gates: [rz, rzz]             # which gates carry disorder (cross-grid check)
      seed_list: [0, 1, 2]                  # the gate instances; averaged
      noise_configs: [noiseless]            # add device_calibrated for the noisy arm
  calibrations:
    - examples/q50_calibration_20260524_08c3c70f.json
```

Field-by-field against the original's globals:

- `fixed.num_qubits: 4` ← the original's `num_qubits` (the example uses 4 for a
  fast demo; the banked reference is 10).
- `fixed.epsilon: 0.03` ← the original's `epsilon = 0.03`.
- `grid.num_kicks: {range: [0, 6]}` ← `for n_kicks in range(num_max_kicks)`,
  stop-exclusive (`0..5`). The banked reference uses `[0, 60]` → `0..59`.
- `disorder_gates: [rz, rzz]` ← tells the cross-grid check that disorder lives
  in `rz` (`hz_angles`) and `rzz` (`Jzz_angles`), not the `rx` drive.
- `seed_list: [0, 1, 2]` ← `num_gate_instances` (3 here vs 10 in the original);
  HPCQC averages the per-seed autocorrelators, the original's
  `autocorrelators / num_gate_instances`.
- `noise_configs: [noiseless]` ← swap/add `device_calibrated` to run the real
  Q50-calibrated noisy arm. `device_calibrated` is opt-in by name (it is **not**
  included in `noise_configs: all`, which means the synthetic-channel tiers).

## Step 4 — run it on LUMI

```bash
sbatch --account=<proj> --partition=standard --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=8 --time=00:30:00 --job-name=floquet_byo \
  --wrap='cd "$SLURM_SUBMIT_DIR" && source "$SLURM_SUBMIT_DIR/env.sh" && \
    export SINGULARITYENV_PYTHONPATH="$HPCQC_ROOT/src" && \
    srun $HPCQC_CPU_WRAPPER $HPCQC_CPU_CONTAINER bash -c \
      "python3 -m lumi_hpc_qc.sweep.run_sweep examples/byo/floquet_byo_sweep.yaml"'
```

HPCQC builds one circuit per `(seed, num_kicks)`, solves placements from the
circuit's own connectivity, runs each under the requested noise, computes the
autocorrelator with the same convention as the original, averages across seeds,
and writes HDF5 + Parquet.

## What you no longer write yourself

The original's MAIN block, FFT, and matplotlib plotting are not part of the
factory — HPCQC stores the autocorrelator vector (and the framework/your own
analysis produces spectra/plots downstream). The factory is **only** the circuit
build; everything else is configuration.

## Notes / gotchas

- **Purity is the contract, not a suggestion.** If you copy a script that draws
  RNG inside the build, the cross-grid check will reject it. Move every random
  or stateful input to a `disorder` arg (file or generated per seed).
- **Keyword-only.** Positional parameters or bare `**kwargs` are rejected by the
  signature check. List every parameter explicitly after `*`.
- **`device_calibrated` pins statevector.** You cannot set `method:
  density_matrix` on it (HPCQC raises with an explanation) — that pin is a
  correctness fix and the scalable path. (See the noise docs / DEBT D2.)
- **Calibration provenance.** The gate-2 reproduction is pinned to the committed
  **05-24** calibration set-id `08c3c70f`; a newer calibration in `examples/` is
  a *different* run, not the reference.

*Reference files (all committed): `examples/byo/floquet_dtc.py`,
`examples/byo/floquet_byo_sweep.yaml`, `examples/byo/floquet_disorder_q4.json`.
Original: `Floquet_DTC_AK7.py`.*

---

## Step 5 — Scaling to the q10 production configuration

The Step 1–4 walkthrough builds a q4/6-kick/3-seed demo to keep iteration fast.
The canonical reference ensemble that the banked
`examples/reference/floquet_dtc_q10_*` artifacts come from is q10 / 60 kicks /
40 seeds / 1000 shots / `master_seed = 0` / polarized initial state. Scaling
the demo to that shape is a pure config-and-data edit — the factory is
unchanged.

Two new files do this:

- **`examples/byo/floquet_disorder_q10.json`** — 40 seeds × 10-qubit disorder,
  generated deterministically from `np.random.SeedSequence(0).spawn(s+1)[s]`
  with `pcg64`, matching the seam the sweep engine's `source: generate` path
  uses (`src/lumi_hpc_qc/sweep/byo_sweep.py::_spawn_rng`). Storing the values
  as data means the sweep is reproducible regardless of any future changes to
  the seeding code; the `_meta.note` field in the JSON records the exact
  reconstruction snippet for re-derivation.
- **`examples/byo/floquet_dtc_q10_sweep.yaml`** — production-scale config:
  `num_qubits: 10`, `num_kicks: {range: [0, 60]}`, 40 seeds, 1000 shots,
  polarized init, and **both `noiseless` and `device_calibrated` in
  `noise_configs:`** so a single sweep produces both arms in one job.

### What stays the same

The factory (`examples/byo/floquet_dtc.py`), the observable (the engine's
`get_autocorrelation`), the placement solver, the aggregator — nothing moves.
The q4 demo and the q10 production config are the same machinery driven by
different YAML.

### Disorder convention preserved

`Jzz_angles` is sampled at length `num_qubits` even though
`build_circuit` only consumes the first `num_qubits - 1` entries (one per
nearest-neighbor bond). The trailing entry is kept to match the q4 example's
shape and AK7's seed-advance pattern. The unused trailing draw doesn't affect
physics; the cross-grid disorder-identity check covers it correctly because it
inspects the materialized `rz`/`rzz` gates in the built circuit, not the
disorder vectors themselves.

### Two arms in one sweep

`noise_configs: [noiseless, device_calibrated]` runs both arms over the same
`(seed, num_kicks)` grid. The engine writes the results to separate output
groups, so the post-hoc analysis can read both without re-running. If you
want only one arm — e.g., to fast-iterate the noiseless path — comment out the
other line; the disorder/seed/shot/kick budget is unchanged.

### Running it

The invocation is the same as Step 4, just pointing at the q10 YAML:

```bash
sbatch --account=project_462001289 --partition=standard --nodes=1 --ntasks-per-node=1 \
  --cpus-per-task=128 --time=01:00:00 --job-name=floquet_q10_byo \
  --wrap='cd "$SLURM_SUBMIT_DIR" && source "$SLURM_SUBMIT_DIR/env.sh" && \
    export SINGULARITYENV_PYTHONPATH="$HPCQC_ROOT/src" && \
    srun $HPCQC_CPU_WRAPPER $HPCQC_CPU_CONTAINER bash -c \
      "python3 -m lumi_hpc_qc.sweep.run_sweep examples/byo/floquet_dtc_q10_sweep.yaml"'
```

`--cpus-per-task=128` matches the canonical 40-instance reference budget (one
worker per instance, well under 128 cores) and `--time=01:00:00` gives generous
headroom over the ~30-min noiseless run for the device-calibrated arm.

### Relationship to the floquet_runner.py reference path

The banked references (`examples/reference/floquet_dtc_q10_noiseless_agg.dat`
and `examples/reference/floquet_dtc_q10_device-cal_agg.dat`) were produced via
`floquet_runner.py` + `aggregate_floquet.py`, **not** the BYO sweep machinery.
The two paths share the same physics (rx → rz → rzz Floquet period, polarized
init, autocorrelator observable) but use different per-seed RNG mechanics:
`floquet_runner.py` uses `np.random.seed(resolve_instance_seed(0, i))` on the
legacy global stream; the BYO sweep uses `pcg64` per-seed Generators spawned
from `SeedSequence(0)`. The resulting disorder values differ numerically per
seed, so the **aggregated autocorrelator from this YAML will not be
byte-identical to the banked `aggregated_autocorr.dat`** — but the ensemble
statistics (mean DTC autocorrelator, period-2 alternating-sign signature,
sem bands) match within shot noise.

The BYO sweep path additionally goes through `prepare_simulation` and the
placement solver (one of HPCQC's selling points), so it produces richer
provenance: per-seed manifests, placement metadata, and the device-calibrated
arm exercises the F5a placement-keyed noise on the same circuits the noiseless
arm runs. Use this YAML when you want the production-grade BYO path with
placement-resolved noise; use `slurm_floquet_40i_60k_1000s.sh` when you want
bit-identical reproduction of the banked references.

### Adapting for a future researcher script (AK8, AK11, …)

The same six-step pattern (this doc, §0–4 + this §5) applies to any future
DTC variant from the same family. Only three things change in practice:

1. **The Floquet-period kernel** (the body of `apply_one_floquet_period`) —
   copy the new researcher script's gate sequence into the factory's
   inner-loop, line-for-line. Anything random or stateful in the new kernel
   moves into the `disorder` block; only deterministic `fixed`/`grid` axes stay
   on the function signature.
2. **The observable** — if the new variant changes `get_autocorrelation` (e.g.,
   AK10's sign convention, or an echo-based observable), the BYO factory contract
   accepts a per-experiment observable function pointer (see DEBT D6/D7 for the
   echo + multi-observable factory extension). Until that lands, the engine's
   `get_autocorrelation` matches AK7 byte-for-byte; verify the new variant's
   formula reduces to it (or extend the contract per D7).
3. **The disorder JSON** — regenerate with the same `pcg64` + `SeedSequence`
   pattern but using the new variant's distribution / cardinality. The `_meta`
   block should record the exact reconstruction snippet so the file is
   self-describing.

The factory module, the YAML schema, the sbatch wrapper, and the aggregation
pipeline are unchanged.
