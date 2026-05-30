#!/usr/bin/env bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
#
# bank_evidence.sh — curate one job's primary sources into evidence/W1/ so a
# reviewer (Team Red) can `git pull` and see exactly what the run produced,
# instead of receiving a console paste or digging through the ~28 MB slurm_logs/
# scratch swamp.
#
# What it does, given a SLURM job id and a short purpose tag:
#   1. Copies slurm_logs/<name>.o<jobid> (and .e if non-trivial) into
#      evidence/W1/<purpose>/ renamed slurm-<jobid>-<purpose>.out (the
#      established naming; root-anchored .gitignore /slurm-*.out does NOT
#      shadow evidence/, so these are tracked).
#   2. Appends the job's sacct row to evidence/W1/sacct_W1_jobs.psv in the
#      exact 16-field pipe-separated schema the file already uses.
#   3. Copies any extra files passed as positional args (e.g. the per-instance
#      .dat the run produced, a z_comb report CSV) into evidence/W1/<purpose>/.
#   4. git-adds evidence/ so the result is staged (does NOT commit/push — the
#      operator reviews and commits, per the Mac-does-all-git workflow).
#
# Usage:
#   scripts/bank_evidence.sh <jobid> <purpose> [extra_file ...]
#
# Examples:
#   scripts/bank_evidence.sh 18938950 d5-multiwave \
#       sweep_output/w1_canary/byo_dat/floquet/QB11/device_calibrated/instance_00_autocorr.dat
#   scripts/bank_evidence.sh 18938652 topology-probe
#   scripts/bank_evidence.sh 18938939 unit-gate-55
#
# Run from the repo root on LUMI (where slurm_logs/ and sacct live). It is
# read-only with respect to slurm_logs/ — it copies, never moves or deletes.

set -euo pipefail

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <jobid> <purpose> [extra_file ...]" >&2
    echo "  <purpose> is a short kebab-case tag, e.g. d5-multiwave, topology-probe" >&2
    exit 2
fi

JOBID="$1"
PURPOSE="$2"
shift 2
EXTRAS=("$@")

# Resolve repo root from this script's location (scripts/ is at repo top level).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EVID_DIR="${REPO_ROOT}/evidence/W1/${PURPOSE}"
SACCT_PSV="${REPO_ROOT}/evidence/W1/sacct_W1_jobs.psv"
SLURM_LOGS="${REPO_ROOT}/slurm_logs"

mkdir -p "${EVID_DIR}"

# ── 1. Copy the slurm stdout/stderr ──────────────────────────────────────────
# Logs land in slurm_logs/<jobname>.o<jobid> OR (for --wrap jobs) at repo root
# as slurm-<jobid>.out. Search both; prefer the named slurm_logs/ form.
found_log=""
for cand in "${SLURM_LOGS}"/*."o${JOBID}" "${REPO_ROOT}/slurm-${JOBID}.out"; do
    if [[ -f "${cand}" ]]; then
        cp "${cand}" "${EVID_DIR}/slurm-${JOBID}-${PURPOSE}.out"
        found_log="${cand}"
        echo "  banked log : ${cand}  ->  evidence/W1/${PURPOSE}/slurm-${JOBID}-${PURPOSE}.out"
        break
    fi
done
if [[ -z "${found_log}" ]]; then
    echo "  [WARN] no stdout log found for job ${JOBID} in slurm_logs/ or repo root" >&2
fi

# stderr only if it carries more than the boilerplate Lmod line (skip the
# 77-byte "replacing craype" noise that every job emits).
for cand in "${SLURM_LOGS}"/*."e${JOBID}"; do
    if [[ -f "${cand}" ]]; then
        # strip the known Lmod boilerplate + blank lines; bank only if anything remains.
        if grep -qvE '^(Lmod is automatically replacing.*|[[:space:]]*)$' "${cand}"; then
            cp "${cand}" "${EVID_DIR}/slurm-${JOBID}-${PURPOSE}.err"
            echo "  banked err : ${cand}  (non-trivial stderr)"
        fi
        break
    fi
done

# ── 2. Append the sacct row in the existing 16-field schema ──────────────────
SACCT_FMT="JobID,JobName,State,Submit,Start,End,Elapsed,ElapsedRaw,MaxRSS,AveRSS,MaxVMSize,AveVMSize,TRESUsageInMax,TRESUsageInTot,ExitCode,Reason"
if command -v sacct >/dev/null 2>&1; then
    if [[ ! -f "${SACCT_PSV}" ]]; then
        # Recreate the header if the file is somehow absent.
        echo "${SACCT_FMT//,/|}" > "${SACCT_PSV}"
    fi
    # -P pipe-delimited, -n no header (we keep the file's single header).
    if ! grep -q "^${JOBID}|" "${SACCT_PSV}"; then
        sacct -j "${JOBID}" -P -n --format="${SACCT_FMT}" --delimiter='|' >> "${SACCT_PSV}"
        echo "  appended   : sacct rows for ${JOBID} -> evidence/W1/sacct_W1_jobs.psv"
    else
        echo "  sacct      : ${JOBID} already present in sacct_W1_jobs.psv (skipped)"
    fi
else
    echo "  [WARN] sacct not on PATH; skipped accounting row for ${JOBID}" >&2
fi

# ── 3. Copy any extra artifacts (dats, report CSVs, scontrol dumps) ──────────
for f in "${EXTRAS[@]:-}"; do
    [[ -z "${f}" ]] && continue
    if [[ -f "${f}" ]]; then
        cp "${f}" "${EVID_DIR}/"
        echo "  banked file: ${f}  ->  evidence/W1/${PURPOSE}/$(basename "${f}")"
    else
        echo "  [WARN] extra file not found: ${f}" >&2
    fi
done

# ── 4. Stage (do NOT commit — operator reviews and commits) ──────────────────
git -C "${REPO_ROOT}" add "evidence/W1/${PURPOSE}" "${SACCT_PSV}" 2>/dev/null || true
echo ""
echo "  Staged evidence/W1/${PURPOSE}/ and the sacct row."
echo "  Review with: git -C '${REPO_ROOT}' status && git -C '${REPO_ROOT}' diff --cached --stat"
echo "  Then commit + push (Mac or LUMI, per your git workflow)."
