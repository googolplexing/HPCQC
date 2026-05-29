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

# LUMI standard partition is SMT-2 (256 logical / 128 physical). The job's
# affinity set is assumed to contain both SMT siblings per owned physical core
# (true when SLURM allocates whole physical cores). Under-counting cores only
# shrinks the cap (under-packs, safe); the memory term is the real OOM guard.
# DEBT: detect siblings precisely rather than assuming a flat /2.
SMT_FACTOR = 2

# D3 headroom: reserve = max(20 GiB floor, 12% fraction). The floor protects
# small allocations; the fraction protects large ones (on a 224 GiB node the
# fraction binds: reserve ~= 26.9 GiB). The reserve must cover OS + page cache
# AND the resident parent heap (the parent holds the imported stack, the full
# work_units list, and every WorkerResult as it returns).
SAFE_MEM_FLOOR_BYTES = 20 * GIB
SAFE_MEM_FRACTION = 0.12


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

    ``safe_mem = limit - max(20 GiB, 0.12 * limit)``. Returns
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

    reserve = max(SAFE_MEM_FLOOR_BYTES, int(SAFE_MEM_FRACTION * limit))
    return limit - reserve, source, warned


def resolve_usable_cores_physical(
    *,
    env: dict | None = None,
    affinity_reader=None,
) -> int:
    """Physical cores the JOB owns.

    ``len(os.sched_getaffinity(0))`` (logical cores in the job's cpuset) halved
    for SMT-2; falls back to ``$SLURM_CPUS_PER_TASK`` / ``$SLURM_CPUS_ON_NODE``,
    then ``os.cpu_count()``. Floored at 1 — a genuine single-core reservation
    resolves to 1 (D4(d): silent/normal, NOT the D10 forced-serial raise).
    """
    env = os.environ if env is None else env
    logical = None
    if affinity_reader is not None:
        logical = affinity_reader()
    else:
        try:
            logical = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            logical = None
    if not logical:
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
    return max(1, logical // SMT_FACTOR)


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
    if safe_mem_bytes is None or safe_mem_bytes <= 0:
        raise ForcedSerialError(
            "no usable memory budget could be resolved (safe_mem unavailable); "
            "refusing to size the pool blind. Set --mem or run under a cgroup "
            "memory limit."
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
