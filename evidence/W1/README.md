# evidence/W1/ — banked W1 verification artifacts

This directory is the **curated, tracked** evidence corpus for the W1 work
(forkserver BYO worker, allocation-aware cap, multi-wave dispatch, z_comb gate).
It is what Team Red reads when verifying: pull the repo, read these files, see
exactly what each run produced — no console paste, no access to the operator's
machine required.

## What belongs here (and what does NOT)

**Here (tracked):** the primary sources for a verifiable claim —
- `slurm-<jobid>-<purpose>.out` — the stdout of a specific run (cap footer,
  SWEEP COMPLETE, verifier verdict).
- `sacct_W1_jobs.psv` — one pipe-separated accounting row per banked job
  (the wall/CPU/MaxRSS/ExitCode record). Single header, append-only.
- `scontrol_partition_<name>.txt`, `scontrol_node_<nid>.txt` — partition and
  node topology dumps that ground a hardware claim (core counts, RealMemory).
- `*_autocorr.dat` — the per-instance or aggregated observable series a run
  produced, when a byte-match / z_comb claim depends on it.
- `<purpose>/` subdirs group the artifacts of one milestone (e.g.
  `gate2_canary/`, `d5-multiwave/`).

**NOT here:** the `slurm_logs/` scratch directory at the repo root. That is
transient development exhaust (hundreds of re-runs, empty stderrs, multi-MB
debug dumps) and is gitignored. Only the handful of logs that *evidence a
specific ruling* get curated into here. If a file is not tied to a claim Red
needs to verify, it does not belong in `evidence/`.

## .gitignore interaction (why these files survive)

The repo `.gitignore` ignores `slurm_logs/` entirely and uses **root-anchored**
patterns `/slurm-*.out` and `/*.patch` so that loose files dropped in the repo
root are ignored — WITHOUT shadowing this directory. `evidence/W1/slurm-*.out`
is several levels deep, so the root-anchored pattern does not match it. This is
deliberate: it lets the operator drop transient logs at the root freely while
keeping the curated corpus tracked. Do not "fix" the `.gitignore` to a
non-anchored `slurm-*.out` — that would silently stop tracking this corpus.

## Banking a run (the one-command habit)

Use the helper rather than hand-copying, so the naming and the sacct schema
stay consistent:

```
scripts/bank_evidence.sh <jobid> <purpose> [extra_file ...]
```

It copies the job's slurm log into `evidence/W1/<purpose>/slurm-<jobid>-<purpose>.out`,
appends the sacct row to `sacct_W1_jobs.psv`, copies any extra artifacts
(dats, z_comb report CSVs), and `git add`s the result. It does **not** commit
or push — the operator reviews (`git diff --cached --stat`) and commits, per
the workflow where git operations are done deliberately.

Examples:
```
scripts/bank_evidence.sh 18938950 d5-multiwave \
    sweep_output/w1_canary/byo_dat/floquet/QB11/device_calibrated/instance_00_autocorr.dat
scripts/bank_evidence.sh 18938652 topology-probe
scripts/bank_evidence.sh 18938939 unit-gate-55
```

## Current corpus (index)

| Artifact | Evidences |
|----------|-----------|
| `slurm-18898399-canary.out` | early W1.3 canary |
| `slurm-18899724-oom.out` | the historical OOM the cap exists to prevent |
| `gate2_canary/sha256_oracle.txt` + `canary_seed_*_autocorr.dat` | the byte-match oracle |
| `runner_*_aggregated_autocorr.dat` | runner reference series (z_comb baseline) |
| `sha256_runner_vs_reference.txt` | runner-vs-reference SHA agreement |
| `scontrol_partition_standard.txt`, `scontrol_node_*.txt` | node topology (core/mem claims) |
| `sacct_W1_jobs.psv` | accounting rows for all banked jobs |

Keep this index current as runs are banked.
