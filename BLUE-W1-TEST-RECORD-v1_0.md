# TEAM BLUE — W1 TEST-RECORD (D-DOC-1)

**Document ID:** BLUE-W1-TEST-RECORD-v1.0
**Date:** May 30, 2026
**Branch / HEAD:** `feature/device-calibrated-noise` @ `6bf022b`
**Purpose:** The curated test-record for the W1 work (forkserver BYO worker,
allocation-aware cap, multi-wave dispatch, the W1.6 z_comb gate, and the gate-2
placement-provenance reconciliation). This is the **D-DOC-1** artifact owed
before W1.6 sign-off: it indexes the W1 corpus + the runs, and presents the
**D-DOC-2(b) ordered-dependency chain** that explains why the gate was
re-specified. Banked primary sources live under `evidence/W1/`; this doc points
to them — it does not restate their numbers from memory.

**Status:** LIVING. W1.1–W1.5 and the gate-2 root-cause + Phase-1 + Step-1
tooling rows are CLOSED; the Step-1 run, reference re-baseline, and Option-1
gate rows are PENDING (Red has approved the path; see §3).

---

## §1 — The W1.6 ordered-dependency chain (D-DOC-2(b))

The spine of the reconciliation, in dependency order. Each row names the
evidence a reviewer reads to verify it.

| # | Step | Verdict / state | Primary evidence |
|---|---|---|---|
| 1 | **W1.6 40-seed production gate-2** (job 18943612, `standard`, 24.4 min) | **FAIL** — worst z 46.34, 52/60 kicks > 5σ; cap footer `cap=80; 1 wave; usable_cores_physical=128` live | `evidence/W1/gate2-fail-18943612/` (slurm `.out`+`.err`, `aggregated_autocorr.dat`, `w1_gate_z_comb_report.csv`); sacct rows in `sacct_W1_jobs.psv` |
| 2 | **Root cause** — qubit-set provenance mismatch, T2-driven | CONFIRMED (Blue + independent Red recompute: T2 2.40× swapped / 1.44× full; runner set reproduced exactly) | `BLUE-FINDING-W1_6-GATE2-PLACEMENT-PROVENANCE-v1.0`; `RED-RESP-GATE2-…-RULING` §2 |
| 3 | **PLACEMENT-1 Phase-1 seam** — researcher `physical_qubits` (solver bypass + list-wise fail-loud + device-cal guardrail) | LANDED `d3f1b3b`; CONFIRMED 18/18 on real deps (job 18946498); Red §6 conditions 1–4 verified | `tests/unit/test_byo_manual_placement.py`; `BLUE-PROPOSAL-RESEARCHER-PLACEMENT-CONTROL-v1.0`; DEBT `PLACEMENT-1`; `RED-APPROVAL-STEP1-CEILING-AND-PHASE1-VERIFY` §3 |
| 4 | **Step-1 verifier mode + ceiling pre-registration** | LANDED `1bbe5da`; numbers APPROVED (floor 0.02, ceiling 2%) | `tests/_w1_z_comb_verify.py --mode step1-residual`; `tests/unit/test_w1_step1_residual.py`; `BLUE-PREREG-STEP1-CONVERGENCE-CEILING-v1.0`; `RED-CLARIFICATION-STEP1-SIGMA-SYS-ORDERING`; `RED-APPROVAL-…` §1–§2 |
| 5 | **Step-1 confirmatory run** — sweep-BYO pinned to runner qubits `QB1,2,5,6,7,9,10,11,12,13`, cal `08c3c70f`, 40 seeds; judged vs the §4 ceiling | **PENDING** — launcher being prepared; reports σ_sys (max/mean rel-dev, depth-trend, decay-rate rel-diff) | (to bank as `evidence/W1/step1-…/` on completion) |
| 6 | **Reference re-baseline on the placement qubits** (Option 1) — regenerate the runner reference on `QB11,5,6,7,13,21,29,28,27,26`, with the placement banked as provenance | **PENDING** — conditioned on row 5 converging | (to bank) |
| 7 | **Option-1 gate of record** — augmented-z `z = |Δ|/√(ref_sem²+cand_sem²+σ_sys²)`, σ_sys from row 5; second identical-qubit residual confirms σ_sys transferred | **PENDING** — Red rules readiness from row 5's residual | (to bank) |

**Decision gate between rows 5 and 6/7:** if Step 1's residual exceeds the
pre-committed 2% ceiling, **stop** — open the idle-implementation finding
(runner ALAP self-scheduling vs sweep `PadDelay`+`RelaxationNoisePass`); do not
widen the ceiling or σ_sys (`RED-CLARIFICATION-STEP1` §5).

## §2 — Why the gate was re-specified (the lesson this record captures)

The gate did its job: it caught that two **physically different** experiments
were being compared (runner reference on fidelity-self-selected qubits vs
sweep-BYO on the solver's `top_1`, only 5/10 overlapping, ~2.4× T2 on the
swapped qubits). The defect was not in the engine, harness, cap fix, or PadDelay
fix — all verified sound — but in the comparison's **basis** (a common
placement) and its **acceptance bar** (a statistical 5σ is a category error for
a cross-*pipeline* systematic; a sub-percent offset reads as tens of σ). The
fix is structural: a placement **contract** (PLACEMENT-1, row 3) so the
reference and candidate can never silently diverge on qubit basis again, plus a
systematic-tolerance bar (rows 4–7). This is the F5 latent gap Red flagged in
session one, now closed by contract rather than patched per-run.

## §3 — Pre-registration (recorded BEFORE the Step-1 run, per RED-CLARIFICATION-STEP1 §2)

The Step-1 convergence call is made against these PRE-COMMITTED numbers, fixed
before the data exists and **distinct from σ_sys** (the ceiling is the defect
line; σ_sys is the measured residual; ceiling ≥ σ_sys):

- **`floor = 0.02`** — `rel_dev = |cand − ref| / max(|ref|, floor)`; below ~0.02
  the autocorrelator is at the noise level and a relative metric is meaningless.
- **Ceiling: `max per-kick rel-dev ≤ 2.0%`** over above-floor kicks, with the
  depth-trend and decay-rate diagnostics consistent with a flat, non-compounding
  offset. Physical expectation on identical qubits is sub-percent; 2% is
  headroom. Red confirmed it would have rejected the real gate-2 failure 16–100×
  over (kick-10 32%, kick-40 126%, kick-58 200%).

**Approved:** `RED-APPROVAL-STEP1-CEILING-AND-PHASE1-VERIFY-v1.0` §1 (both
numbers), verified at `1bbe5da`. Mechanism: `tests/_w1_z_comb_verify.py
--mode step1-residual --floor 0.02 --max-rel-dev 0.02` (exit 0 converged / 1
over-ceiling / 3 structural), cross-checked by `tests/unit/test_w1_step1_residual.py`.
**σ_sys** (the measured residual) is carried forward to row 7, preferring the
relative form so it transfers between the runner qubits and the placement qubits.

## §4 — W1 corpus index

**Banked evidence (`evidence/W1/`):**

| Subdir / file | What it evidences |
|---|---|
| `gate2-fail-18943612/` | The W1.6 gate-2 FAIL (row 1) — fix-provenance (D-DOC-3) |
| `unit-gate-55/` | The in-container unit gate (regression baseline) |
| `d5-multiwave/` | Multi-wave dispatch (both arms) + allocation-aware cap |
| `w1_3-canary-clean/`, `w1_3-canary-first/` | W1.3 BYO worker canaries |
| `topology-probe/` | Node/partition topology (the sibling-map / core-count grounding) |
| `gate2_canary/` | Pre-production gate-2 canary |
| `runner_*_aggregated_autocorr.dat`, `sha256_runner_vs_reference.txt` | Runner reference provenance |
| `scontrol_*` | Partition/node hardware facts (RealMemory, core counts) |
| `sacct_W1_jobs.psv` | Accounting rows for every banked W1 job |

**Key correspondence (chain docs):** `BLUE-FINDING-W1_6-GATE2-PLACEMENT-PROVENANCE`,
`BLUE-PROPOSAL-RESEARCHER-PLACEMENT-CONTROL`, `RED-RESP-GATE2-…-RULING`,
`RED-CLARIFICATION-STEP1-SIGMA-SYS-ORDERING`, `BLUE-PREREG-STEP1-CONVERGENCE-CEILING`,
`RED-APPROVAL-STEP1-CEILING-AND-PHASE1-VERIFY`; debt anchor `DEBT.md` → `PLACEMENT-1`.

**Code anchors:** `tests/slurm_w1_gate.sh` (gate harness), `tests/_w1_z_comb_verify.py`
(gate + step1-residual verifier), `src/lumi_hpc_qc/sweep/{sweep_engine,placement_solver}.py`
(the `physical_qubits` seam), `scripts/bank_evidence.sh` (banking helper).

---

*Living document. On each pending row's completion, bank the run under
`evidence/W1/<purpose>/` and fill its evidence cell here, so the chain stays the
single ordered-dependency view of W1.6 through sign-off.*
