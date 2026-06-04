# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""W1.4 — allocation-aware forkserver worker cap.

Pure and dependency-light (stdlib only — no qiskit / aer / h5py) so the cap
arithmetic and its resolution/fallback chains are unit-testable with mocked
cgroup / affinity / probe inputs on the cheap on-stack gate, with no
node-exclusive reservation. Wired into ``SweepEngine._execute_byo_group``,
replacing the W1.3 placeholder cap.

Implements RED-RESP-W1.3-VERIFY-AND-W1.4-CAP-RULINGS-v1.0 D1–D6 and the Q6
formula:

    cap = min( cpu_workers,              # config ceiling (SweepConfig.cpu_workers, default 128)
               num_units,
               usable_cores_physical,    # cores the JOB owns (affinity / SLURM), halved for SMT-2
               floor(safe_mem / per_unit_peak) )   # memory bound, from the ALLOCATION

Both resource terms read the job's runtime allocation, never node ``scontrol``.

Design notes
------------
* The cap sizes every worker at its FULL ``VmHWM`` (D6 conservative): there is
  no preloaded-heap CoW (NF5), so a worker carries its full resident footprint
  post-fork. Over-counting shared ``.so`` pages -> under-packs -> never OOMs.
* D2 conservative-now: the whole pool is sized by the ``device_calibrated``
  per-unit peak (all units treated as heavy). The device-cal/noiseless pool
  split is DEBT.
* D4 fail-loud is raised here, not swallowed (the D10 forced-serial signal).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

GIB = 1024 ** 3

# C1 banked fallback (RED-RESP-W1 Q6: "1-unit probe over a banked constant").
# Used as per_unit_peak ONLY when no device_calibrated unit exists to probe
# live (e.g. a noiseless-only sweep) or the probe yielded no VmHWM. Conservative
# single-unit device-cal figure consistent with the §1.4 corpus. Always
# provenance-tagged in the footer when it fires (never a silent default).
C1_PER_UNIT_PEAK_BYTES = 3 * GIB

# Physical-core counting reads the actual CPU topology (sibling map) rather
# than assuming an SMT factor. The job's affinity set is the logical CPUs it
# owns; the number of *physical* cores is the count of DISTINCT
# thread-sibling groups among them. This is correct for any allocation shape:
#   - one thread per core (the normal SLURM allocation, e.g. --cpus-per-task=128
#     -> logical 0-127, siblings 0,128 / 1,129 / ... -> 128 distinct -> 128 cores)
#   - both threads of some cores (e.g. --threads-per-core=2, or a 0-255 set ->
#     logical 0 and 128 share sibling-list "0,128" and collapse to one core)
# Verified on LUMI-C at d057e64: --cpus-per-task=128 -> 128 distinct cores
# (job 18938652), where the prior flat //2 wrongly reported 64. The //2 was a
# systematic 2x under-count wherever SLURM allocates one thread per core (the
# common case), under-packing the pool on both `standard` and `small`.
#
# FALLBACK_SMT_FACTOR applies ONLY when the topology is unreadable (no
# /sys/devices/system/cpu, e.g. some restricted containers) AND we have only a
# bare logical count from SLURM env / os.cpu_count(). In that last-resort path
# we cannot know the sibling layout, so we conservatively divide (under-count
# -> under-pack -> safe, never OOM). The primary path never divides.
FALLBACK_SMT_FACTOR = 2

# D3 headroom: reserve = min( max(16 GiB floor, 12% fraction), 50% cap ).
# The reserve must cover OS + page cache AND the resident parent heap (the
# parent holds the imported stack, the full work_units list, and every
# WorkerResult as it returns).
#
# Three bounds, because a single fixed floor is wrong at both ends (RED-RESP-
# W1-CAP-VERIFY-AND-GATE-RULING §2.3, D3): the 16 GiB absolute floor protects
# a *large* allocation from a too-small percentage reserve; the 12% fraction
# scales the reserve up on big nodes (224 GiB -> 26.9 GiB); and the 50% cap
# stops the floor from *over*-reserving a SMALL allocation. Without the cap, a
# fixed floor sabotages exactly the partial-node allocations it was meant to
# protect: on the non-exclusive `small` partition a researcher can request a
# few cores and a proportional --mem (e.g. 16 GiB), and a 16-or-20 GiB floor
# would reserve >= the whole allocation -> safe_mem <= 0 -> a spurious
# ForcedSerialError on a job that should run. The cap makes the reserve at
# most half of whatever was allocated, so a small slice still runs (16 GiB
# --mem -> reserve 8 -> safe_mem 8), while the node-exclusive 224 GiB case is
# unchanged (12% = 26.9 GiB < 50% = 112 GiB, so the cap does not bind there).
SAFE_MEM_FLOOR_BYTES = 16 * GIB
SAFE_MEM_FRACTION = 0.12
# Upper bound on the reserve as a fraction of the resolved limit, so the floor
# can never swallow a small allocation (D3 / §2.3).
SAFE_MEM_RESERVE_CAP_FRACTION = 0.5


class ForcedSerialError(RuntimeError):
    """D4(a)/(b): the allocation cannot be honored. Raised, never silent.

    (a) one unit's projected peak exceeds safe_mem (cannot run even one unit);
    (b) the cap collapses to 1 on a multi-core reservation BECAUSE the memory
        term forced it (the D10 forced-serial signal).
    """


# ── Probe (D1) ─────────────────────────────────────────────────────────────

def read_vmhwm_kib(pid: str | int = "self") -> int | None:
    """Kernel high-water mark ``VmHWM`` (KiB) from ``/proc/<pid>/status``.

    This is the EXACT peak (D1) — the kernel's high-water mark captures the
    build/transpile spike — NOT a sampled MaxRSS (``jobacct_gather`` samples on
    a ~30 s cadence and under-reports: the OOM job sampled 17.7 GiB while the
    true peak crossed 224 GiB). Falls back to ``getrusage`` if /proc is
    unreadable. Returns KiB, or None if neither source is available.
    """
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])  # already KiB
    except (OSError, ValueError, IndexError):
        pass
    try:
        import resource
        # ru_maxrss is KiB on Linux (bytes on macOS, but the worker runs on
        # Linux/LUMI). Only used as a fallback when /proc is unavailable.
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None


# ── Resource resolution (D3 / cores) ─────────────────────────────────────────

def _read_int_file(path: str) -> int | None:
    """Read a single integer from a cgroup/sysfs file.

    ``"max"`` (cgroup v2 unlimited) and any non-int -> None (do NOT parse "max"
    as a number, D3). Missing/unreadable -> None.
    """
    try:
        with open(path) as f:
            raw = f.read().strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _read_meminfo_total_bytes(path: str = "/proc/meminfo") -> int | None:
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024  # kB -> bytes
    except (OSError, ValueError, IndexError):
        return None
    return None


def resolve_cgroup_limit_bytes(
    v2_path: str = "/sys/fs/cgroup/memory.max",
    v1_path: str = "/sys/fs/cgroup/memory/memory.limit_in_bytes",
) -> tuple[int | None, str | None]:
    """cgroup memory bound: v2 ``memory.max`` -> v1 ``memory.limit_in_bytes``.

    Returns (bytes, source) or (None, None). ``"max"`` / missing -> fall
    through (None). LUMI Slurm is cgroup v2 but v1 is read as a fallback so we
    don't hard-assume (D3).
    """
    v = _read_int_file(v2_path)
    if v is not None:
        return v, "cgroup_v2:memory.max"
    v = _read_int_file(v1_path)
    if v is not None:
        return v, "cgroup_v1:memory.limit_in_bytes"
    return None, None


def resolve_safe_mem(
    *,
    env: dict | None = None,
    cgroup_reader=resolve_cgroup_limit_bytes,
    meminfo_reader=_read_meminfo_total_bytes,
) -> tuple[int | None, str, bool]:
    """Resolve the memory budget. Returns ``(safe_mem_bytes, source, warned)``.

    Fallback chain (D3 ratified): cgroup memory.max(v2)/limit_in_bytes(v1) ->
    SLURM (``$SLURM_MEM_PER_NODE`` in MB / ``--mem``) -> node RealMemory
    (``/proc/meminfo`` MemTotal). ``warned`` is True when we fall through to
    node RealMemory — the non-allocation-aware last resort Q6 exists to avoid;
    the caller emits a LOUD WARN in the footer (never silent).

    ``safe_mem = limit - min(max(16 GiB, 0.12*limit), 0.5*limit)``. Returns
    ``(None, "unavailable", True)`` if nothing is readable (forces a raise
    downstream rather than silently over-packing).
    """
    env = os.environ if env is None else env
    node_total = meminfo_reader()

    limit, source = cgroup_reader()
    # A cgroup limit larger than the node is the v1 "unlimited" sentinel
    # (~INT64) or otherwise not a real per-job bound -> fall through.
    if limit is not None and node_total is not None and limit > node_total:
        limit, source = None, None

    warned = False
    if limit is None:
        slurm_mb = env.get("SLURM_MEM_PER_NODE")
        if slurm_mb:
            try:
                limit = int(slurm_mb) * 1024 * 1024  # MB -> bytes
                source = "slurm:SLURM_MEM_PER_NODE"
            except ValueError:
                limit = None
    if limit is None and node_total is not None:
        limit = node_total
        source = "node:MemTotal(RealMemory)"
        warned = True
    if limit is None:
        return None, "unavailable", True

    # D3 reserve: at least max(16 GiB, 12%), but never more than 50% of the
    # allocation, so a small --mem slice is not over-reserved into a spurious
    # ForcedSerialError (RED-RESP-W1-CAP-VERIFY-AND-GATE-RULING §2.3).
    reserve = max(SAFE_MEM_FLOOR_BYTES, int(SAFE_MEM_FRACTION * limit))
    reserve = min(reserve, int(SAFE_MEM_RESERVE_CAP_FRACTION * limit))
    return limit - reserve, source, warned


def _read_thread_siblings(cpu: int) -> str | None:
    """The thread-sibling group of one logical CPU, as the kernel reports it.

    Reads ``/sys/devices/system/cpu/cpu<N>/topology/thread_siblings_list`` — a
    stable string like ``"0,128"`` that is IDENTICAL for both threads of a
    physical core. Distinct strings => distinct physical cores. Returns None if
    the topology file is unreadable (restricted container / non-Linux).
    """
    path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def resolve_usable_cores_physical(
    *,
    env: dict | None = None,
    affinity_reader=None,
    siblings_reader=_read_thread_siblings,
) -> int:
    """Physical cores the JOB owns.

    Counts DISTINCT physical cores in the job's affinity set by deduplicating
    each logical CPU's thread-sibling group (NOT a flat SMT divide — see the
    FALLBACK_SMT_FACTOR note). Correct for any allocation shape: one thread per
    core, or both threads of some cores (they share a sibling group and collapse
    to one). Floored at 1 — a genuine single-core reservation resolves to 1
    (D4(d): silent/normal, NOT the D10 forced-serial raise).

    Resolution order:
      1. Affinity set + sibling map (the accurate path; needs /sys topology).
      2. Affinity set with NO readable topology -> conservative divide of the
         set size by FALLBACK_SMT_FACTOR (last resort; under-counts, safe).
      3. No affinity -> SLURM env count / os.cpu_count(), divided (last resort).

    ``affinity_reader`` returns the SET of logical CPU IDs the job owns (for
    tests); ``siblings_reader(cpu)`` returns that CPU's sibling-group string or
    None (for tests). Both default to the real OS/sysfs readers.
    """
    env = os.environ if env is None else env

    # (1)/(2): the affinity set — the logical CPUs this job actually owns.
    affinity: set[int] | None
    if affinity_reader is not None:
        affinity = affinity_reader()
    else:
        try:
            affinity = set(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            affinity = None

    if affinity:
        groups = {sib for c in affinity if (sib := siblings_reader(c)) is not None}
        if groups:
            # (1) Accurate: distinct sibling groups == distinct physical cores.
            return max(1, len(groups))
        # (2) Topology unreadable but we DO know the owned-logical count.
        # Cannot know the sibling layout -> divide conservatively (under-count).
        return max(1, len(affinity) // FALLBACK_SMT_FACTOR)

    # (3) No affinity at all -> bare logical count from SLURM env, then cpu_count.
    logical = None
    for key in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE"):
        val = env.get(key)
        if val:
            try:
                logical = int(val)
                break
            except ValueError:
                pass
    if not logical:
        logical = os.cpu_count() or 1
    return max(1, logical // FALLBACK_SMT_FACTOR)


# ── The cap (Q6 formula + D4 taxonomy) ───────────────────────────────────────

@dataclass
class CapDecision:
    """Resolved cap + the terms behind it (criterion-5 observability)."""
    cap: int
    cpu_workers: int
    num_units: int
    usable_cores_physical: int
    mem_term: int                 # floor(safe_mem / per_unit_peak)
    safe_mem_bytes: int
    per_unit_peak_bytes: int
    binding_term: str             # 'cpu_workers' | 'num_units' | 'cores' | 'memory'


def compute_worker_cap(
    *,
    cpu_workers: int,
    num_units: int,
    usable_cores_physical: int,
    safe_mem_bytes: int | None,
    per_unit_peak_bytes: int,
) -> CapDecision:
    """Resolve the allocation-aware cap (Q6 formula + D4 fail-loud taxonomy).

    PURE over scalars (mockable). Raises ``ForcedSerialError`` for D4(a)/(b);
    returns a CapDecision otherwise.

    D4 taxonomy:
      (a) one unit's projected peak > safe_mem      -> raise (cannot run at all)
      (b) cap collapses to 1 on a multi-core reservation BECAUSE memory forced
          it (num_units>1 AND min(cpu_workers,cores)>1 AND mem term == 1)
                                                     -> raise (D10 forced-serial)
      (c) cap == 1 solely because num_units == 1     -> silent, normal
      (d) cap == 1 because usable_cores_physical == 1 -> silent, normal (NOT (b))
    """
    if per_unit_peak_bytes <= 0:
        raise ValueError("per_unit_peak_bytes must be > 0")
    if num_units < 1:
        raise ValueError("num_units must be >= 1")
    if safe_mem_bytes is None:
        raise ForcedSerialError(
            "no usable memory budget could be resolved (safe_mem unavailable); "
            "refusing to size the pool blind. Set --mem or run under a cgroup "
            "memory limit."
        )
    if safe_mem_bytes <= 0:
        raise ForcedSerialError(
            f"resolved memory budget is non-positive (safe_mem="
            f"{safe_mem_bytes / GIB:.2f} GiB): the headroom reserve met or "
            f"exceeded the allocation. This should not happen via resolve_safe_mem "
            f"(its reserve is capped at 50% of the limit), so the allocation "
            f"itself is too small for even the reserve. Request more memory "
            f"(--mem) for this job."
        )

    mem_term = safe_mem_bytes // per_unit_peak_bytes

    # D4(a): a single unit does not fit at all.
    if mem_term < 1:
        raise ForcedSerialError(
            f"D4(a): projected per-unit peak {per_unit_peak_bytes / GIB:.2f} GiB "
            f"exceeds safe_mem {safe_mem_bytes / GIB:.2f} GiB — cannot run even "
            f"one unit on this allocation."
        )

    core_ceiling = min(cpu_workers, usable_cores_physical)
    cap = max(1, min(cpu_workers, num_units, usable_cores_physical, mem_term))

    # D4(b): memory forced a multi-core reservation down to serial. Distinct
    # from (c) num_units==1 and (d) usable_cores_physical==1 (core_ceiling==1).
    if cap == 1 and num_units > 1 and core_ceiling > 1 and mem_term == 1:
        raise ForcedSerialError(
            f"D4(b)/D10 forced-serial: worker cap collapsed to 1 on a "
            f"{usable_cores_physical}-physical-core reservation "
            f"(cpu_workers={cpu_workers}, num_units={num_units}) because the "
            f"memory term forced it: safe_mem {safe_mem_bytes / GIB:.2f} GiB / "
            f"per_unit_peak {per_unit_peak_bytes / GIB:.2f} GiB = {mem_term}. "
            f"The reservation has cores to spare but not memory."
        )

    # Provenance: which term bound the cap. Ties resolve by this fixed order.
    terms = (
        ("cpu_workers", cpu_workers),
        ("num_units", num_units),
        ("cores", usable_cores_physical),
        ("memory", mem_term),
    )
    binding_term = min(terms, key=lambda kv: kv[1])[0]

    return CapDecision(
        cap=cap,
        cpu_workers=cpu_workers,
        num_units=num_units,
        usable_cores_physical=usable_cores_physical,
        mem_term=mem_term,
        safe_mem_bytes=safe_mem_bytes,
        per_unit_peak_bytes=per_unit_peak_bytes,
        binding_term=binding_term,
    )


# ── RED-DIRECTIVE-PROBE-SKIP-WHEN-NON-BINDING ───────────────────────────────
# The D1/D2 probe exists ONLY to learn per_unit_peak -> mem_term. mem_term
# changes the cap iff it is below core_units_ceiling = min(cpu_workers,
# num_units, usable_cores_physical) (the per_unit_peak-independent part of the
# cap). When the engine can prove, with a CONSERVATIVE upper bound peak_hi >=
# true_peak, that memory cannot bind even at that pessimistic peak, the probe's
# result cannot change the cap, so it is skipped. peak_hi is engine-owned and
# validated once against the §1.4 corpus (soundness gate) — no researcher input.

# Generous multiple of the working-state copies aer holds (state + temporaries).
# Conservative on purpose: a too-large factor only makes the engine probe more
# often (lost speedup), never under-pack -> never an OOM. It is material only
# where state_bytes approaches FIXED_OVERHEAD (statevector ~n>=26, density_matrix
# ~n>=13) — the large-n regime where memory binds and the engine probes anyway.
PEAK_HI_STATE_FACTOR = 4


def _state_bytes(method: str, n: int) -> int:
    """Bytes of the dominant simulator state for ONE unit at ``n`` qubits.

    ``statevector`` holds one complex128 amplitude per basis state -> ``16*2**n``
    (device_calibrated pins statevector). ``density_matrix`` holds a complex128
    entry per (row, col) -> ``16*2**(2n)``. An unknown method is treated as the
    heavier density_matrix size (conservative — peak_hi must be an upper bound).
    """
    if method == "statevector":
        return 16 * (1 << n)
    return 16 * (1 << (2 * n))


def conservative_peak_hi(
    methods, n: int, *, state_factor: int = PEAK_HI_STATE_FACTOR
) -> int:
    """A validated UPPER bound on one device-cal unit's VmHWM for this run.

    ``peak_hi = C1_PER_UNIT_PEAK_BYTES + state_factor * max_method_state_bytes``.
    The fixed-overhead term reuses the conservative C1 banked figure (resident
    stack + noise model + COW), which already dominates the §1.4 corpus at small
    n; the state term only becomes material at large n. ``methods`` is the set of
    NoiseConfig methods present in the group; the bound takes the heaviest. The
    soundness obligation (peak_hi >= every measured VmHWM over the corpus, the
    heaviest observable family included) is enforced by the soundness gate, not
    assumed here.
    """
    method_set = tuple(methods) if methods else ("statevector",)
    state_hi = max(_state_bytes(m, n) for m in method_set)
    return C1_PER_UNIT_PEAK_BYTES + state_factor * state_hi


@dataclass(frozen=True)
class ProbeSkipDecision:
    """Result of the non-binding probe-skip test (RED-DIRECTIVE-PROBE-SKIP)."""
    skip: bool
    core_units_ceiling: int
    peak_hi_bytes: int
    peak_source: str | None   # "skip:mem_non_binding" when skip, else None


def decide_probe_skip(
    *,
    cpu_workers: int,
    num_units: int,
    usable_cores_physical: int,
    safe_mem_bytes: int | None,
    peak_hi_bytes: int,
) -> ProbeSkipDecision:
    """Decide whether the D1/D2 probe can be skipped (memory provably non-binding).

    PURE over scalars (mockable, like ``compute_worker_cap``). Skip iff
    ``safe_mem // peak_hi >= core_units_ceiling`` where ``core_units_ceiling =
    min(cpu_workers, num_units, usable_cores_physical)``.

    OOM-safety: ``safe_mem // peak_hi >= core_units_ceiling`` implies
    ``safe_mem >= core_units_ceiling * peak_hi``. When skipping, the cap is
    ``core_units_ceiling``, so at most that many units run concurrently, each
    using ``<= true_peak <= peak_hi``; total concurrent memory
    ``<= core_units_ceiling * peak_hi <= safe_mem`` — the allocation provably
    holds the packed units. A larger ``peak_hi`` only shrinks ``mem_term`` and
    so makes the test HARDER to pass (probe more), never the reverse.

    ``safe_mem`` unavailable -> do NOT skip (sizing on an unknown budget is
    forbidden; the caller falls through to the probe and the D4 raise).
    """
    core_units_ceiling = min(cpu_workers, num_units, usable_cores_physical)
    skip = (
        safe_mem_bytes is not None
        and peak_hi_bytes > 0
        and safe_mem_bytes // peak_hi_bytes >= core_units_ceiling
    )
    return ProbeSkipDecision(
        skip=skip,
        core_units_ceiling=core_units_ceiling,
        peak_hi_bytes=peak_hi_bytes,
        peak_source="skip:mem_non_binding" if skip else None,
    )


# RED-VERIFY-PROBE-SKIP-CLOSURE §2 condition (b): the runtime skip is gated to
# (method, n) shapes whose peak_hi bound has been validated against a MEASURED
# VmHWM corpus (the soundness gate). A skip never fires at an n whose bound was
# never measured — at an unvalidated shape the engine falls through to the probe,
# so OOM-safety never rests on the peak_hi FORMULA alone, only on a measured
# bound. Extend this set (AND the soundness corpus that backs it) when a new
# (method, n) is measured.
CORPUS_VALIDATED_PEAK_HI_SHAPES: frozenset[tuple[str, int]] = frozenset({
    ("statevector", 10),   # §1.4 corpus: max measured 1.32 GiB << peak_hi ~3 GiB
})


def peak_hi_corpus_validated(methods, n: int) -> bool:
    """Whether peak_hi(method, n) is corpus-validated for EVERY method present.

    PURE. True iff every method's ``(method, n)`` is in
    ``CORPUS_VALIDATED_PEAK_HI_SHAPES`` — i.e. peak_hi was checked against a
    measured VmHWM there. Empty ``methods`` defaults to statevector (matching
    ``conservative_peak_hi``). The runtime skip ANDs this in, so an unvalidated
    shape falls through to the probe rather than skipping on a never-measured
    bound (RED-VERIFY-PROBE-SKIP-CLOSURE §2 condition b).
    """
    method_set = tuple(methods) if methods else ("statevector",)
    return all((m, n) in CORPUS_VALIDATED_PEAK_HI_SHAPES for m in method_set)


# MEASURED per-unit VmHWM ceilings (bytes) for corpus-validated shapes. The
# value is the MAX measured VmHWM over the §1.4 corpus for that (method, n),
# heaviest observable family included — i.e. a measured upper bound on one
# unit's true peak, NOT the conservative C1 formula. Source of truth is the
# soundness corpus (_MEASURED_VMHWM_CORPUS in tests/unit/test_probe_skip_non_binding.py
# / RED-RESP-W1-CAP-VERIFY §1.4); the (method, n) keys MUST stay identical to
# CORPUS_VALIDATED_PEAK_HI_SHAPES (asserted in the unit test) so a shape can
# never be skip-eligible without a measured peak to size the cap on.
#
# Used INSTEAD OF conservative_peak_hi for the probe-skip test (and the skipped
# cap) on a corpus-validated shape: the conservative C1 bound (~3 GiB) exists
# only because an UNMEASURED unit's peak is unknown; once a shape is measured,
# its measured max is the correct, tighter upper bound, and using it lets a
# provably non-binding run skip the probe AT FULL concurrency rather than under-
# packing to the conservative mem_term. OOM-safety is unchanged: decide_probe_skip
# proves `core_units_ceiling * bound <= safe_mem`, and `bound >= true_peak` holds
# here because the value is the measured MAX times CORPUS_PEAK_SAFETY_FACTOR (>= 1).
CORPUS_MEASURED_PEAK_BYTES: dict[tuple[str, int], int] = {
    ("statevector", 10): int(1.32 * GIB),  # §1.4: max measured (autocorr arm)
}

# Margin applied to the measured corpus peak before it is used as the bound.
# Red-owned safety knob: 1.0 uses the measured max directly (the D3 safe_mem
# reserve already covers OS / page-cache / parent-heap headroom, so the per-unit
# value need not be inflated again); raise it to add per-unit variance headroom
# at the cost of pushing some runs back onto the probe. Documented and ruled in
# BLUE-TO-RED-CORPUS-PEAK-PROBE-SKIP.
CORPUS_PEAK_SAFETY_FACTOR = 1.0


def corpus_measured_peak_hi(methods, n: int) -> int | None:
    """Measured per-unit peak bound (bytes) for this shape, or None if unmeasured.

    PURE. Returns ``max measured VmHWM over the present methods * factor`` iff
    EVERY method present has a ``(method, n)`` entry in
    ``CORPUS_MEASURED_PEAK_BYTES`` (same "every method present" semantics as
    ``peak_hi_corpus_validated``); otherwise None, so the caller falls back to
    ``conservative_peak_hi``. Empty ``methods`` defaults to statevector.

    The returned bound is a valid upper bound on one unit's true peak (measured
    max * factor>=1) and is <= ``conservative_peak_hi`` for the same shape — a
    tighter-but-still-sound bound that the probe-skip can size on.
    """
    method_set = tuple(methods) if methods else ("statevector",)
    if not all((m, n) in CORPUS_MEASURED_PEAK_BYTES for m in method_set):
        return None
    base = max(CORPUS_MEASURED_PEAK_BYTES[(m, n)] for m in method_set)
    return int(base * CORPUS_PEAK_SAFETY_FACTOR)
