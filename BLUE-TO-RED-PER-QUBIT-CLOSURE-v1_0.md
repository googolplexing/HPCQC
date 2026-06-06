# BLUE → RED — Closure: per-qubit (site-resolved) autocorrelator — conditions 1–4 satisfied, gates GREEN, unit suite GREEN

**From:** Team Blue · **Re:** acceptance of the per-qubit autocorrelator un-collapse per `RED-RULING-PER-QUBIT-AUTOCORRELATOR-AND-SITE-RESOLVED-RUN-v1.0` (APPROVED IN PRINCIPLE). All four conditions are implemented and verified in-container; requesting final acceptance.
**Tree:** per-qubit series (P1 observable+serializer, P2 producer wiring, P3 merge reducer + consumer reader + gates + DEBT) landed at `45d8fe4`; test relocation into `tests/unit/` at `2cb5dd0` (test-move only — the engine code at `45d8fe4` and `2cb5dd0` is identical). Base `3b9a91f`.
**Status:** The scalar path is byte-identical and the dataset name is unchanged; the per-qubit output is additive. Gate logs and `sacct` rows are banked under `evidence/W1/perqubit-gates/` and `evidence/W1/perqubit-units/`; job IDs inline for re-verification.

---

## §1 — The four conditions, each satisfied, with the artifact that proves it

| Condition (ruling) | How it is met | Evidence |
|---|---|---|
| **1 — self-describing per-qubit `.dat`** (`kick local_q physical_q mean sem`, D1) **+ a consumer-side ordering test** proving the reader maps `local_q → physical_qubit_set[i]` | `data.persite_output.write_persite_series` emits the five-column self-describing form; `byo_observable.aggregate_byo_autocorr_perqubit` pins the D1 columns. The reader attributes each site by the file's `physical_q` column, **not** a path re-parse. | Unit job `19082957`: `test_persite_output::test_self_describing_and_ordered`, `test_byo_autocorr_perqubit::test_perqubit_dat_is_self_describing_and_ordered`, and the **end-to-end** `test_map_dtc_per_qubit_ordering::test_reader_attributes_signal_to_correct_physical_qubit` (a period-2 signal on a deliberately non-monotonic placement lands on the correct physical qubit, not a neighbor). 11/11 passed. |
| **2 — the §6/§5.4 equivalence gate covers per-qubit output dedup-on vs dedup-off via the path-keyed comparator** | The dedup gate's certified comparators are generic: `_assert_dats_identical` byte-compares **every** `.dat` (now including `aggregated_autocorr_perqubit.dat`) and `_assert_h5_subtree_equal` compares **every** `/byo` dataset (now including `autocorrelator_perqubit`). A presence assertion was added so the per-qubit coverage cannot be vacuous. | Gate job `19082764`, `GATE4_RC=0`: `off=16 on=12` (dedup **engages** — 4 non-canonical noiseless units skipped) with `.dat` + `/byo` output byte-identical through the broadcast/re-stamp. |
| **3 — dataset name stays `autocorrelator`; legacy scalar `.dat` byte-identical (never dropped)** | `get_autocorrelation` (scalar) is untouched; P2/P3 are additive (new sibling dataset `autocorrelator_perqubit`, new `.dat`). The reducer recovers the scalar series unchanged, so `aggregated_autocorr.dat` and the `autocorrelator` dataset are byte-for-byte preserved. | Gate job `19082764`, `GATE2_RC=0` (`GATE PASSED [byo]`): single-node vs 2- vs 3-rank merge byte-identity over the full `.dat` tree + `/byo` subtree — which includes the unchanged scalar artifacts alongside the new per-qubit ones. |
| **4 — `map_dtc_to_qpu_3d.py` FAILS LOUD when per-qubit is requested but absent; DEBT entry opened** | `per_qubit_dtc(per_qubit=True)` raises `SystemExit` rather than smear the chain scalar as per-site; default (`per_qubit=False`) keeps the legacy chain-resolution surface. `DEBT.md` D12 records the closed gap and the by-design fail-loud residual. | Unit job `19082957`: `test_map_dtc_per_qubit_ordering::test_reader_fails_loud_when_perqubit_requested_but_absent` and `::test_legacy_smear_path_unchanged`. DEBT D12 in tree. |

## §2 — The two gates, green and non-vacuous on the field each guards

- **GATE 2 (fan-out byte-identity, single vs 2- vs 3-rank), `GATE2_RC=0`** — exercises the P3 **merge reducer**: the merged `aggregated_autocorr_perqubit.dat` and the `autocorrelator_perqubit` dataset are byte-identical to the single-node engine path across allocation shapes. This is the in-engine proof of the reducer code (which `py_compile` + unit tests could not exercise end-to-end — the standing "green ≠ runs" lesson).
- **GATE 4 (§5.4 noiseless dedup, off vs on), `GATE4_RC=0`** — `off=16 on=12` proves dedup engages; byte-identical output proves the §2.2 broadcast is safe; the presence assertion proves the per-qubit artifacts are actually present in both arms (non-vacuous).
- **Unit suite, job `19082957`** — 11/11 passed under the default import mode from `tests/unit/`, covering P1 (serializer + observable) and P3 (consumer reader). The per-qubit-mean-equals-legacy-scalar invariant (`mean over wires == get_autocorrelation`) is checked to 1e-12.

`SUMMARY: gate2(flat)=0  gate4(dedup)=0 → ALL GATES PASSED`.

## §3 — Scope and disclosures (carry all)

1. **D2 honored:** both the scalar and the per-qubit forms are emitted on every BYO autocorrelator run — the per-qubit output never replaces the scalar.
2. **D5 honored:** only the autocorrelator is un-collapsed. Bare polarization ⟨Z_i(t)⟩ remains a separate observable, out of this scope.
3. **Fail-loud residual is by design (DEBT D12):** a true site-resolved Q50 surface requires a run with the per-qubit observable emitted; the existing scalar-only survey (job `19064225`) can only produce the chain-smeared surface, and `--per-qubit` will refuse to fake per-site from it. The reader docstring's prior over-claim of a per-qubit branch is removed now that the branch exists.
4. **D3 ε decision is still pending the physicist** and gates only the *run config*, not these patches. On that decision the ε run (Fig 2 e–f + partial per-site surface) proceeds on the F5a HIGH/LOW pair (D4); no further engine change is needed.

## §4 — What acceptance unblocks

The per-site data is the central scientific payoff — it discriminates coherence-limited vs MBL-protected DTC per site, enables edge-vs-bulk and weak-link diagnosis, and gives a stronger noise-model validation target (per-site pattern vs hardware). On acceptance: (a) the small-ε run on the F5a HIGH/LOW pair once D3 lands, producing Fig 2(e,f) and the partial per-site surface via `map_dtc_to_qpu_3d.py --per-qubit`; (b) optionally a re-run of the chain survey with the per-qubit observable ON (free in compute — same counts; the un-collapse is a reduction choice) for the full per-site Q50 surface.

Requesting: acceptance of conditions 1–4 as satisfied.

*Effort ID:* `RED-RULING-PER-QUBIT-AUTOCORRELATOR-AND-SITE-RESOLVED-RUN`.
