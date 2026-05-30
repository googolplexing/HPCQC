# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""W1.4 — allocation-aware worker cap (worker_cap.py).

Runs on the cheap on-stack gate (no node-exclusive reservation): the cap
arithmetic + D3 fallback chains are pure/stdlib and exercised with injected
cgroup/affinity/probe values, so no real /sys or SLURM allocation is needed.

Coverage maps to the rulings:
  * D5 binding condition — production arithmetic (cap == 66) + historical-OOM
    (job 18899724 shape: caps DOWN below the old placeholder of 80, and the
    raise variant when a unit exceeds safe_mem).
  * D4 fail-loud taxonomy — (a) unit > safe_mem raises; (b) memory-forced
    serial on a multi-core reservation raises; (c) num_units==1 silent;
    (d) single physical core silent (NOT a raise).
  * D3 safe_mem — cgroup v2 / v1 / "max" fall-through / SLURM env / node
    RealMemory (LOUD-WARN flag) / unavailable.
  * usable_cores_physical — SMT-2 halving, floor 1, SLURM-env fallback.
  * D1 — read_vmhwm_kib parses VmHWM and ignores cgroup "max".
"""
from __future__ import annotations

import sys

import pytest

from lumi_hpc_qc.sweep import worker_cap as wc

G = wc.GIB


# ── D5: production arithmetic ────────────────────────────────────────────────

def test_production_arithmetic_cap_is_66():
    # safe_mem ~= 200 GiB, per_unit_peak ~= 3 GiB, 80 units, 128 cores/workers.
    d = wc.compute_worker_cap(
        cpu_workers=128, num_units=80, usable_cores_physical=128,
        safe_mem_bytes=200 * G, per_unit_peak_bytes=3 * G,
    )
    assert d.cap == 66           # floor(200 / 3) = 66 ; two waves over 80 units
    assert d.mem_term == 66
    assert d.binding_term == "memory"


# ── D5: historical-OOM (job 18899724 shape) ─────────────────────────────────

def test_historical_oom_caps_down_below_placeholder():
    # The old placeholder cap = min(num_units, os.cpu_count()) launched all 80
    # units -> 80 * 3 = 240 GiB > safe_mem -> OOM. The allocation-aware cap must
    # cap DOWN so the in-flight set fits.
    placeholder = min(80, 256)
    d = wc.compute_worker_cap(
        cpu_workers=128, num_units=80, usable_cores_physical=128,
        safe_mem_bytes=200 * G, per_unit_peak_bytes=3 * G,
    )
    assert d.cap < placeholder
    assert d.cap * 3 <= 200            # in-flight set fits safe_mem
    assert placeholder * 3 > 200       # the placeholder would have OOM'd


def test_historical_oom_raises_when_unit_exceeds_safe_mem():
    # If even a single unit's probed peak exceeds safe_mem, D4(a) raises rather
    # than silently launching an unrunnable pool.
    with pytest.raises(wc.ForcedSerialError):
        wc.compute_worker_cap(
            cpu_workers=128, num_units=80, usable_cores_physical=128,
            safe_mem_bytes=200 * G, per_unit_peak_bytes=210 * G,
        )


# ── D4 fail-loud taxonomy ────────────────────────────────────────────────────

def test_d4a_unit_exceeds_safe_mem_raises():
    with pytest.raises(wc.ForcedSerialError, match="D4.a."):
        wc.compute_worker_cap(
            cpu_workers=128, num_units=4, usable_cores_physical=128,
            safe_mem_bytes=2 * G, per_unit_peak_bytes=3 * G,
        )


def test_d4b_memory_forced_serial_on_multicore_raises():
    # mem_term == 1 while cores + workers + units all > 1 -> D10 forced-serial.
    with pytest.raises(wc.ForcedSerialError, match="D4.b./D10"):
        wc.compute_worker_cap(
            cpu_workers=128, num_units=80, usable_cores_physical=128,
            safe_mem_bytes=4 * G, per_unit_peak_bytes=3 * G,
        )


def test_d4c_single_unit_is_silent_cap_one():
    d = wc.compute_worker_cap(
        cpu_workers=128, num_units=1, usable_cores_physical=128,
        safe_mem_bytes=4 * G, per_unit_peak_bytes=3 * G,
    )
    assert d.cap == 1
    assert d.binding_term == "num_units"


def test_d4d_single_core_is_silent_cap_one():
    # cap==1 because the reservation has one physical core — NOT memory-forced.
    d = wc.compute_worker_cap(
        cpu_workers=128, num_units=80, usable_cores_physical=1,
        safe_mem_bytes=200 * G, per_unit_peak_bytes=3 * G,
    )
    assert d.cap == 1
    assert d.binding_term == "cores"


# ── binding-term provenance ──────────────────────────────────────────────────

@pytest.mark.parametrize("kwargs,expected", [
    (dict(cpu_workers=4, num_units=80, usable_cores_physical=128,
          safe_mem_bytes=200 * G, per_unit_peak_bytes=3 * G), "cpu_workers"),
    (dict(cpu_workers=128, num_units=3, usable_cores_physical=128,
          safe_mem_bytes=200 * G, per_unit_peak_bytes=3 * G), "num_units"),
    (dict(cpu_workers=128, num_units=80, usable_cores_physical=8,
          safe_mem_bytes=200 * G, per_unit_peak_bytes=3 * G), "cores"),
    (dict(cpu_workers=128, num_units=80, usable_cores_physical=128,
          safe_mem_bytes=60 * G, per_unit_peak_bytes=3 * G), "memory"),
])
def test_binding_term_provenance(kwargs, expected):
    assert wc.compute_worker_cap(**kwargs).binding_term == expected


def test_compute_cap_rejects_nonpositive_peak():
    with pytest.raises(ValueError):
        wc.compute_worker_cap(
            cpu_workers=128, num_units=4, usable_cores_physical=128,
            safe_mem_bytes=200 * G, per_unit_peak_bytes=0,
        )


# ── D3 safe_mem fallback chain (injected readers) ────────────────────────────

def test_safe_mem_cgroup_v2():
    safe, src, warned = wc.resolve_safe_mem(
        env={},
        cgroup_reader=lambda: (224 * G, "cgroup_v2:memory.max"),
        meminfo_reader=lambda: 224 * G,
    )
    assert src == "cgroup_v2:memory.max"
    assert not warned
    # reserve = min( max(16 GiB, 0.12*224) , 0.5*224 ) = 26.88 GiB (fraction
    # binds; floor and 50% cap do not) -> safe ~= 197.1 GiB. Node-exclusive
    # case is unchanged by the D3 §2.3 reserve cap.
    assert abs(safe - (224 * G - int(0.12 * 224 * G))) < 2
    assert 196 * G < safe < 198 * G


def test_safe_mem_small_allocation_not_over_reserved():
    # D3 §2.3 regression: on the non-exclusive `small` partition a researcher
    # can request a partial node with a proportional --mem. A fixed 16-or-20 GiB
    # floor would reserve >= the whole allocation -> safe_mem <= 0 -> a spurious
    # ForcedSerialError. The 50% reserve cap must keep a small slice runnable.
    # 16 GiB --mem: reserve = min(max(16, 1.92), 8) = 8 GiB -> safe_mem = 8 GiB.
    safe, src, warned = wc.resolve_safe_mem(
        env={"SLURM_MEM_PER_NODE": str(16 * 1024)},  # 16 GiB in MB
        cgroup_reader=lambda: (None, None),
        meminfo_reader=lambda: 992 * G,              # small-partition node total
    )
    assert src == "slurm:SLURM_MEM_PER_NODE"
    assert not warned
    assert safe > 0, "small --mem must not be over-reserved into a raise"
    assert abs(safe - 8 * G) < 2
    # And it must actually size a pool rather than raise: at 1.32 GiB/unit the
    # historical measured peak, an 8 GiB budget supports several workers.
    d = wc.compute_worker_cap(
        cpu_workers=128, num_units=8, usable_cores_physical=4,
        safe_mem_bytes=safe, per_unit_peak_bytes=int(1.32 * G),
    )
    assert d.cap >= 1 and d.mem_term >= 1


def test_safe_mem_reserve_never_exceeds_half():
    # The reserve is capped at 50% of the limit for any allocation size.
    for mem_gib in (4, 8, 16, 24, 32, 64, 128):
        safe, _, _ = wc.resolve_safe_mem(
            env={"SLURM_MEM_PER_NODE": str(mem_gib * 1024)},
            cgroup_reader=lambda: (None, None),
            meminfo_reader=lambda: 992 * G,
        )
        assert safe >= mem_gib * G * 0.5 - 2, f"--mem={mem_gib} over-reserved"
        assert safe > 0


def test_compute_cap_distinguishes_nonpositive_from_unavailable():
    # None (detection failed) and <=0 (too-small budget) must give DIFFERENT,
    # actionable messages so a small-partition user is not misled into thinking
    # memory detection broke (D3 §2.3).
    with pytest.raises(wc.ForcedSerialError, match="unavailable"):
        wc.compute_worker_cap(
            cpu_workers=4, num_units=4, usable_cores_physical=4,
            safe_mem_bytes=None, per_unit_peak_bytes=int(1.32 * G),
        )
    with pytest.raises(wc.ForcedSerialError, match="non-positive"):
        wc.compute_worker_cap(
            cpu_workers=4, num_units=4, usable_cores_physical=4,
            safe_mem_bytes=-(1 * G), per_unit_peak_bytes=int(1.32 * G),
        )


def test_safe_mem_cgroup_v1():
    safe, src, warned = wc.resolve_safe_mem(
        env={},
        cgroup_reader=lambda: (100 * G, "cgroup_v1:memory.limit_in_bytes"),
        meminfo_reader=lambda: 224 * G,
    )
    assert src == "cgroup_v1:memory.limit_in_bytes"
    assert not warned


def test_safe_mem_max_falls_through_to_slurm():
    # cgroup "max" -> reader returns (None, None); SLURM_MEM_PER_NODE (MB) used.
    safe, src, warned = wc.resolve_safe_mem(
        env={"SLURM_MEM_PER_NODE": "229376"},   # 224 GiB in MB
        cgroup_reader=lambda: (None, None),
        meminfo_reader=lambda: 224 * G,
    )
    assert src == "slurm:SLURM_MEM_PER_NODE"
    assert not warned
    assert 196 * G < safe < 198 * G


def test_safe_mem_cgroup_over_node_is_sentinel_falls_through():
    # cgroup v1 "unlimited" sentinel (> node total) must not be taken literally.
    safe, src, warned = wc.resolve_safe_mem(
        env={},
        cgroup_reader=lambda: (2 ** 62, "cgroup_v1:memory.limit_in_bytes"),
        meminfo_reader=lambda: 224 * G,
    )
    assert src == "node:MemTotal(RealMemory)"
    assert warned


def test_safe_mem_realmemory_warns():
    safe, src, warned = wc.resolve_safe_mem(
        env={},
        cgroup_reader=lambda: (None, None),
        meminfo_reader=lambda: 224 * G,
    )
    assert src == "node:MemTotal(RealMemory)"
    assert warned is True


def test_safe_mem_unavailable_returns_none():
    safe, src, warned = wc.resolve_safe_mem(
        env={},
        cgroup_reader=lambda: (None, None),
        meminfo_reader=lambda: None,
    )
    assert safe is None
    assert src == "unavailable"
    assert warned


def test_compute_cap_raises_on_unavailable_safe_mem():
    with pytest.raises(wc.ForcedSerialError):
        wc.compute_worker_cap(
            cpu_workers=128, num_units=80, usable_cores_physical=128,
            safe_mem_bytes=None, per_unit_peak_bytes=3 * G,
        )


# ── usable_cores_physical (SMT halving / floor / fallback) ───────────────────

# ── usable_cores_physical: distinct-physical-core counting via sibling map ───
# A fake LUMI-C topology: 128 physical cores, SMT-2, siblings (N, N+128).
def _lumi_siblings(cpu):
    lo = cpu % 128
    return f"{lo},{lo + 128}"


def test_usable_cores_production_128_not_64():
    # MEASURED on LUMI-C (job 18938652): --cpus-per-task=128 -> affinity 0-127,
    # which is the first thread of all 128 distinct physical cores. The correct
    # count is 128 (the old flat //2 wrongly said 64, under-packing the gate).
    affinity = set(range(128))
    assert wc.resolve_usable_cores_physical(
        affinity_reader=lambda: affinity, siblings_reader=_lumi_siblings) == 128


def test_usable_cores_both_siblings_collapse_to_core():
    # Both threads of every core present (e.g. --threads-per-core=2, or a 0-255
    # set): logical 0 and 128 share sibling group "0,128" and must collapse to
    # ONE core. 256 logical -> 128 physical, NOT 256.
    affinity = set(range(256))
    assert wc.resolve_usable_cores_physical(
        affinity_reader=lambda: affinity, siblings_reader=_lumi_siblings) == 128


def test_usable_cores_small_partition_partial():
    # MEASURED on `small` (job 18937495): --cpus-per-task=4 -> affinity like
    # [82,83,84,85], four DISTINCT cores' first threads -> 4 physical cores.
    # The old //2 wrongly said 2 (under-pack). This is also why the post-fix
    # D5 wave-forcing recipe uses --cpus-per-task=2, not 4.
    affinity = {82, 83, 84, 85}
    assert wc.resolve_usable_cores_physical(
        affinity_reader=lambda: affinity, siblings_reader=_lumi_siblings) == 4


def test_usable_cores_single_core_floor_one():
    # A genuine single-core reservation floors at 1 (D4(d)).
    assert wc.resolve_usable_cores_physical(
        affinity_reader=lambda: {5}, siblings_reader=_lumi_siblings) == 1


def test_usable_cores_topology_unreadable_falls_back_to_divide():
    # /sys topology unreadable (restricted container): we know the owned-logical
    # count but not the sibling layout, so divide conservatively (under-count,
    # safe). 128 owned logical -> 64.
    affinity = set(range(128))
    assert wc.resolve_usable_cores_physical(
        affinity_reader=lambda: affinity, siblings_reader=lambda c: None) == 64


def test_usable_cores_no_affinity_slurm_env_fallback():
    # No affinity at all -> SLURM env count, divided conservatively.
    assert wc.resolve_usable_cores_physical(
        env={"SLURM_CPUS_ON_NODE": "256"}, affinity_reader=lambda: None) == 128


def test_usable_cores_no_affinity_empty_set_uses_env():
    # An empty affinity set is treated as "no affinity" -> env fallback.
    assert wc.resolve_usable_cores_physical(
        env={"SLURM_CPUS_PER_TASK": "8"}, affinity_reader=lambda: set()) == 4


# ── D1 probe parsing ─────────────────────────────────────────────────────────

@pytest.mark.skipif(not sys.platform.startswith("linux"),
                    reason="VmHWM via /proc is Linux-only")
def test_read_vmhwm_self_positive():
    val = wc.read_vmhwm_kib("self")
    assert val is not None and val > 0


def test_read_int_file_ignores_max(tmp_path):
    p = tmp_path / "memory.max"
    p.write_text("max\n")
    assert wc._read_int_file(str(p)) is None
    p.write_text("123456789\n")
    assert wc._read_int_file(str(p)) == 123456789
    assert wc._read_int_file(str(tmp_path / "does_not_exist")) is None
