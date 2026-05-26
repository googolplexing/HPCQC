# BLUE-DESIGN — D3.4: BYO Execution Path (counts → autocorrelator under device-calibrated noise)

**Document ID:** BLUE-DESIGN-D3.4-BYO-EXECUTION-v1.0
**Date:** 2026-05-26
**From:** Team Blue
**To:** Team Red
**Re:** RED-SIGNOFF gate 3 (D3.4) + the prerequisite for gate 2 (F4 bit-exact reproduction). Follows D3.1 (DEBT re-scope, `771cbc0`), D3.2 (F5a seam, `26d0beb`), D3.3 (route `device_calibrated`, in `a76d121`).
**Status:** Design for review. Grounded in the `a76d121` tree (verified facts marked ✓). No execution code lands until Red signs the autocorrelator convention (§4) and the guardrail/Parquet decisions (§6).
**Verification basis:** `HPCQC-a76d121.tar.gz` read line-by-line.

---

## §1 — Purpose and where this sits

D3.4 is the BYO **execution** branch: it replaces the D3.3 fail-loud `NotImplementedError` in `_execute_group` (✓ `sweep_engine.py`, the `source != "channels"` guard) with the path that actually runs a researcher's parameterized circuit factory under device-calibrated noise and produces the counts→autocorrelator observable. It is the consumer Gap A built up to (`_build_byo_circuit` is staged-but-uncalled, ✓ BLUE-DELIVERY §4) and the prerequisite for gate 2 — the F4 reproduction needs a BYO sweep that actually *runs* a circuit under `device_calibrated`.

This is also the user-facing headline: **researcher supplies a circuit-factory `.py` + a sweep `.yaml`; HPCQC sweeps it under device-calibrated noise.** The factory contract (§3), YAML schema (§5), and autocorrelator convention (§4) here are the spec the `docs/index.html` BYO guide and the `Floquet_DTC_AK7.py` migration example document.

## §2 — What is already in place (✓ verified at `a76d121`)

- **Build seam:** `SweepEngine._build_byo_circuit(task)` (✓ `sweep_engine.py:1941`) → `load_circuit(script_params=assemble_build_kwargs(fixed, disorder, grid))`. Builds an invariant-checked circuit per task; **not yet called by `_execute_group`**.
- **Expansion:** `_expand_byo_experiment` (✓ `:382`), cal × seed (outer) × grid (inner); per-seed disorder resolved once and shared by identity across the grid; default-ON cross-grid check.
- **Connectivity:** `extract_connectivity(qc)` (✓ `circuit_loader.py:268`) → sorted unique `(i,j)`, `i<j`, the circuit's own 2q pattern — the BYO placement boundary (not a `topology_library` entry).
- **Noise routing:** `device_calibrated` is a registered `NoiseConfig` with `source="device_calibrated"`, `method="statevector"`, opt-in by name (✓ D3.3; `"all"` excludes it).
- **Per-placement noise seam (F5a):** `prepare_simulation(..., physical_qubits=, physical_edges=)` threads a placement into `device_noise.build_control_readout_noise_model` + `build_relaxation_pass`, with identity `initial_layout` (✓ D3.2, `26d0beb`; `tests/unit/test_f5a_placement_noise.py` 13/13).
- **Autocorrelator reference:** `floquet_runner_v2.get_autocorrelation(counts, init_bit_array, num_qubits)` (✓ `:94-110`), byte-identical to the original `Floquet_DTC_AK7.py` (✓ `:26` states this). §4 pins D3.4 to **this exact function**.

The gap D3.4 fills: a counts-based execution branch that calls `_build_byo_circuit`, solves placements from `extract_connectivity`, runs each placement × env via `prepare_simulation` for **counts**, computes the autocorrelator with the §4 convention, and stores it with the placement + guardrail flag.

## §3 — The researcher's circuit-factory contract

A BYO factory is a Python module exposing a **pure, keyword-only** build function (the committed `examples/byo/floquet_dtc.py:build_circuit` ✓ is the reference):

1. **Keyword-only, all parameters as data.** `def build_circuit(*, num_kicks, epsilon, num_qubits, hz_angles, Jzz_angles, init_bit_array) -> QuantumCircuit`. `validate_factory_signature` (✓ Gap A) checks the signature against grid ∪ fixed ∪ disorder keys at submit time; positional-only and surprise `**kwargs` are rejected.
2. **Pure — no build-time randomness.** The factory must draw **no** RNG and read no global mutable state. Disorder enters as arguments (`hz_angles`, `Jzz_angles`, `init_bit_array`), resolved once per seed and shared by identity across the grid. This is enforced by the default-ON cross-grid identity check (✓ `extract_disorder_signature`, two-highest comparison): an impure factory that draws fresh RNG per build yields different signatures at two grid points and the sweep **raises** before execution.
3. **The `num_kicks=0` t=0 reference is legal.** A grid that includes `num_kicks=0` builds a circuit with no disorder-bearing gates (empty signature) — the autocorrelator's t=0 normalization point. The two-highest cross-grid comparison (✓ D-A §3, RED-REVIEW §2) exists precisely so this degenerate low point doesn't false-positive.
4. **Derived constants computed internally.** e.g. `h_x = (1-epsilon)·π` is computed in the factory, not supplied — so `epsilon` can be promoted from `fixed` to a grid axis by a YAML-only edit, and the `rx` drive is deliberately excluded from the disorder signature so sweeping `epsilon` doesn't trip the check.

## §4 — The counts → autocorrelator convention (gate-2 binding; REQUIRES Red sign-off)

D3.4's observable helper MUST reproduce `floquet_runner_v2.get_autocorrelation` (✓ `:94-110`) **exactly**, because gate 2 is "bit-exact within shot noise" against the banked CSV that was produced by that function. The convention, verified in-tree and against the original `Floquet_DTC_AK7.py`:

- Per bitstring: reverse to little-endian (`bit_array = list(bitstring)[::-1]` ✓), compare each wire to `init_bit_array[wire]`; `plus` if equal, `minus` if not.
- `temp_corr = (plus - minus) * count`; sum over bitstrings.
- Normalize: `total_corr / (total_shots * num_qubits)` (✓ — `num_qub` is taken from the bitstring length, equals `num_qubits`).
- Per-seed (per "gate instance") autocorrelators are **averaged across seeds**, matching the original `autocorrelators / num_gate_instances` (✓ `aggregate_floquet.py:8-9`) and the committed `.dat` → CSV chain.

**D3.4 will not re-implement this from scratch** — it will call the existing `get_autocorrelation` (or a thin shared helper extracted from it, byte-identical) so there is one definition. Red verifies at delivery (RED-REVIEW §3.4) that the helper matches `aggregate_floquet.py`'s convention exactly (same 0-indexed kick, same estimator). **This is the seam we most want signed before code**, since a convention drift here is exactly what would make gate 2 fail to reproduce.

## §5 — The sweep-YAML schema (researcher-facing)

A parameterized BYO experiment (the committed `examples/byo/floquet_byo_sweep.yaml` ✓ is the reference):

Schema verified against the committed `examples/byo/floquet_byo_sweep.yaml` ✓
(exact keys: `seed_list`, `disorder.initial_state`, experiment-level
`disorder_gates`):

```yaml
sweep:
  experiments:
    - type: byo_circuit
      label: floquet_dtc_example
      circuit_script: examples/byo/floquet_dtc.py   # the factory module
      circuit_function: build_circuit               # the keyword-only build fn
      fixed:                                         # constant kwargs
        num_qubits: 4
        epsilon: 0.03
      grid:                                          # swept axes (one circuit per point)
        num_kicks: {range: [0, 60]}                  # stop-exclusive -> 0..59; includes t=0
      disorder:                                      # per-seed disorder (purity contract)
        source: file                                 # file | generate
        file: examples/byo/floquet_disorder_q4.json  # hz_angles/Jzz_angles/init_bit_array per seed
        initial_state: 3                             # polarized (matches original)
      disorder_gates: [rz, rzz]                      # which gates carry disorder (cross-grid check)
      seed_list: [0, 1, 2]                           # gate instances -> averaged
      noise_configs: [device_calibrated, noiseless]  # opt-in by name (D3.3)
      # placement: top_1                             # see §6 guardrail (engine-set in D3.4)
  calibrations:
    - examples/q50_calibration_20260524_08c3c70f.json  # gate-2 pins THIS set-id
```

Key points: `byo_circuit` is exempt from the `hamiltonians`/`qubit_sizes` requirement (✓ RED §3.1, `validate_sweep_config`); `noise_configs` must request `device_calibrated`/`noiseless` by name (not `all`); the calibration for the gate-2 reproduction is pinned to the committed **05-24** set-id `08c3c70f` (NOT the newer 05-26 file — different run). The committed example uses `range: [0, 6]` and `[noiseless]` for a fast smoke; the gate-2 reproduction uses `[0, 60]` and both noise arms.

## §6 — Placement guardrail + Parquet (RED-REVIEW §4 Q2/Q3)

Until per-placement composition is exercised end-to-end and verified, a multi-placement `device_calibrated` run could *look* placement-resolved while the engine wiring is still settling. Per Red's ruling:

- **`top_1`-default:** D3.4 restricts `device_calibrated` to the single top-fidelity placement unless explicitly overridden. This makes the limitation structural (one placement, no false per-placement dimension).
- **`noise_placement_independent` Parquet column:** `bool`, default `false`; set `true` when a `device_calibrated` run did not resolve noise per placement, and **also record the single `physical_qubit_set` used** so the row is self-describing (✓ Q3). Reconcile against the 36-column benchmark schema in `data/benchmark_export.py` (new from `main`) as an **additive** column, not a rename — D3.4 will confirm this against that file before writing.
- When D3.4 wires the F5a call site (passing each placement's `physical_qubits`/`physical_edges` from the solver into `prepare_simulation`), and gate 2 verifies index-`k` idle relaxation maps to `physical_qubits[k]` on the scheduled circuit, the guardrail is dropped and DEBT D3a closes.

## §7 — Execution branch design

A BYO branch in `_execute_group` (dispatch on `experiment_type == "byo_circuit"`, or a parallel `_execute_byo_group`):

1. **Build** the per-task circuit via `_build_byo_circuit` (§2).
2. **Placements** from `extract_connectivity` on the *built* circuit → the placement solver (`top_1` per §6).
3. **Run for counts** per placement × noise env: `device_calibrated` → `prepare_simulation(source="device-calibrated", physical_qubits=<placement>, physical_edges=<placement edges>, calibration_path=...)` then run for `shots`; `noiseless` → `prepare_simulation(source="noiseless", ...)`, statevector, no noise. Counts, not ⟨H⟩ — this path is separate from the density_matrix twin battery (✓ `_execute_group` is hamiltonian/⟨H⟩-oriented; BYO is counts-based).
4. **Autocorrelator** via the §4 helper per (seed, placement); average across seeds.
5. **Store** to HDF5/Parquet with the placement and the `noise_placement_independent` flag (§6).

**First-increment scope:** `device_calibrated` + `noiseless` only (the example YAML uses exactly these). Synthetic density_matrix tiers under the counts observable are deferred (DEBT D6).

## §8 — Open questions for Red

1. **Autocorrelator helper (§4):** call `floquet_runner_v2.get_autocorrelation` directly, or extract a byte-identical shared helper into `sweep/`? We lean extract-shared (one definition, importable by the engine without pulling the v2 runner), provided it is proven byte-identical.
2. **Guardrail default (§6):** `top_1`-default confirmed? (We implement `top_1` unless overridden.)
3. **Parquet column (§6):** confirm `noise_placement_independent` + `physical_qubit_set` are additive to the `benchmark_export.py` schema; confirm exact column name/type before we write.
4. **Gate-2 pre-registration (RED-REVIEW §3.3):** we will pre-register shots + autocorrelator tolerance (±k·sem from the committed CSV's per-kick `sem`) in the D3.5 delivery **before** the run. Confirm `k` (we propose k=3).

## §9 — Doc + migration deliverables (land with D3.4 code)

- `docs/index.html`: rewrite `#byo-guide` to the two-mode story (fixed/eval-only **and** parameterized factory+YAML sweep), add `device_calibrated`/`source` to `#sweep-yaml` and the noise table, refresh version/roadmap. Parameterized path marked **in progress (D3.4)** until the code lands and verifies, then flipped.
- `examples/byo/MIGRATION_FloquetDTC.md`: the `Floquet_DTC_AK7.py` → `floquet_dtc.py` + `floquet_byo_sweep.yaml` walkthrough, emphasizing the purity refactor (RNG/disorder out of the build into per-seed args — the §3.2 contract) and the instance→seed mapping.

*— Team Blue. The build seam, connectivity, F5a per-placement noise, and `device_calibrated` routing are all in-tree; D3.4 is the counts execution branch that joins them, pinned to the one autocorrelator definition (`floquet_runner_v2:94-110`) that makes gate 2 reproducible. Requesting sign-off on the §4 convention and the §6 guardrail/Parquet decisions before execution code lands.*
