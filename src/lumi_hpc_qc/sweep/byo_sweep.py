# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Gap A scaffold — pure-Python core for parameterized BYO circuit sweeps.

Implements the qiskit-independent logic of SPEC-002-7.5 v1.2 so the risky,
RED-scrutinized parts (grid expansion, the disorder invariant, signature
validation, range sugar) are isolated and unit-tested before wiring into the
qiskit-dependent engine.

INTEGRATION (per §7.5.9 — "no new top-level module"): these functions are
intended to be merged into the existing BYO ingestion / sweep modules:
  * desugar_axis / expand_circuit_grid  -> sweep/circuit_loader.py (grid desugaring)
                                           + sweep/sweep_engine.py expand_grid()
  * validate_factory_signature          -> sweep/circuit_loader.py (load/preview)
  * resolve_disorder                    -> sweep/sweep_engine.py (per-seed seam)
  * assemble_build_kwargs               -> the per-task build seam
  * cross_grid_identity_check           -> the per-task build seam (default-ON)
A small sweep/byo_sweep.py module is an acceptable alternative home if Red
prefers isolation over growing sweep_engine.py; flagged for the doc-hygiene note.

This file has NO qiskit dependency on purpose. The one place a real circuit is
needed (cross_grid_identity_check) takes the factory and a param-extractor as
arguments, so the check logic is testable with a stub and the qiskit-specific
gate-parameter reader is injected by the engine.
"""
from __future__ import annotations

import inspect
import json
from typing import Any, Callable

import numpy as np


# ═══════════════════════════════════════════════════════════════════════
# §7.5.3 — Axis value forms (range sugar; stop-EXCLUSIVE, Q1)
# ═══════════════════════════════════════════════════════════════════════

def desugar_axis(name: str, spec: Any) -> list:
    """Resolve one grid axis to an explicit list (the form recorded in provenance).

    Accepts:
      (a) explicit list      -> returned as-is (ints/strs/floats preserved)
      (b) {"range":[a,b]}    -> list(range(a,b))    stop-EXCLUSIVE
      (c) {"range":[a,b,c]}  -> list(range(a,b,c))  step c

    Validates per §7.5.3 / RED §A1: range bounds and step must be int (a float
    step would crash range() and produce float kicks); reject step==0 and
    empty/degenerate ranges.
    """
    if isinstance(spec, list):
        if len(spec) == 0:
            raise ValueError(f"grid axis '{name}': explicit list is empty")
        return list(spec)

    if isinstance(spec, dict) and "range" in spec:
        rng = spec["range"]
        if not (isinstance(rng, list) and len(rng) in (2, 3)):
            raise ValueError(
                f"grid axis '{name}': range must be [start, stop] or "
                f"[start, stop, step], got {rng!r}"
            )
        # Reject float (and bool) bounds/step: range() is int-only, and a float
        # 'num_kicks' crashes range() downstream (F-5).
        for v in rng:
            if isinstance(v, bool) or not isinstance(v, int):
                raise ValueError(
                    f"grid axis '{name}': range values must be int "
                    f"(stop-exclusive, Python range semantics), got {v!r}"
                )
        if len(rng) == 3 and rng[2] == 0:
            raise ValueError(f"grid axis '{name}': range step must not be 0")
        resolved = list(range(*rng))
        if len(resolved) == 0:
            raise ValueError(
                f"grid axis '{name}': range {rng} is empty "
                f"(start >= stop with positive step, or vice versa)"
            )
        return resolved

    raise ValueError(
        f"grid axis '{name}': unsupported spec {spec!r}; "
        f"use an explicit list or {{range: [start, stop[, step]]}}"
    )


def expand_circuit_grid(grid: dict[str, Any]) -> list[dict]:
    """Cartesian product of the desugared axes, declared-order preserved.

    Returns an ordered list of grid-point dicts. The first declared axis varies
    slowest (stable, readable ordering for a plot's independent variable).
    Empty grid -> [{}] (the degenerate single no-axis circuit, §7.1 behavior).
    """
    if not grid:
        return [{}]
    axes = [(name, desugar_axis(name, spec)) for name, spec in grid.items()]
    points: list[dict] = [{}]
    for name, values in axes:
        points = [{**p, name: v} for p in points for v in values]
    return points


# ═══════════════════════════════════════════════════════════════════════
# §7.5.1 — Engine-validated factory signature (F3 edge cases)
# ═══════════════════════════════════════════════════════════════════════

def validate_factory_signature(
    fn: Callable,
    *,
    grid_keys: set[str],
    fixed_keys: set[str],
    disorder_keys: set[str],
    allow_kwargs: bool = False,
) -> None:
    """Check supplied keys against the factory signature (§7.5.1, V1.1-F3).

    Raises ValueError with a precise message on: a key declared in two blocks;
    a positional-only parameter; an unrecognized supplied key; a missing
    REQUIRED parameter (no default). Parameters WITH defaults are optional.
    A **kwargs factory is rejected unless allow_kwargs=True (signature_check:
    false), which the caller should pair with a loud preview warning.
    """
    # Duplicate keys across blocks.
    for a, b, an, bn in (
        (grid_keys, fixed_keys, "grid", "fixed"),
        (grid_keys, disorder_keys, "grid", "disorder"),
        (fixed_keys, disorder_keys, "fixed", "disorder"),
    ):
        dup = a & b
        if dup:
            raise ValueError(
                f"parameter(s) {sorted(dup)} declared in both '{an}' and '{bn}'; "
                f"each parameter belongs to exactly one block"
            )

    supplied = grid_keys | fixed_keys | disorder_keys

    sig = inspect.signature(fn)
    params = sig.parameters
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    if has_var_kw and not allow_kwargs:
        raise ValueError(
            f"factory '{fn.__name__}' exposes **kwargs, which defeats the "
            f"signature safety check. Remove it, or set signature_check: false "
            f"to opt out (a preview warning will be emitted)."
        )

    positional_only = [
        n for n, p in params.items()
        if p.kind is inspect.Parameter.POSITIONAL_ONLY
    ]
    if positional_only:
        raise ValueError(
            f"factory '{fn.__name__}' has positional-only parameter(s) "
            f"{positional_only} (before '/'); the engine spreads by keyword, "
            f"so make them keyword-only or positional-or-keyword"
        )

    required = {
        n for n, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
        and p.default is inspect.Parameter.empty
    }
    accepted = {
        n for n, p in params.items()
        if p.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                      inspect.Parameter.KEYWORD_ONLY)
    }

    if not has_var_kw:
        unknown = supplied - accepted
        if unknown:
            raise ValueError(
                f"config supplies key(s) {sorted(unknown)} that factory "
                f"'{fn.__name__}' does not accept. Accepts: {sorted(accepted)}"
            )
    missing = required - supplied
    if missing:
        raise ValueError(
            f"factory '{fn.__name__}' requires {sorted(missing)} but no "
            f"grid/fixed/disorder block provides them"
        )


# ═══════════════════════════════════════════════════════════════════════
# §7.5.5 — Disorder data: load (file) or generate, with F1/F2 asserts
# ═══════════════════════════════════════════════════════════════════════

def resolve_disorder(
    disorder_spec: dict,
    seed_values: list[int],
    *,
    num_qubits: int,
    configured_initial_state: int | None = None,
    sampler: Callable[[Any, int], dict] | None = None,
) -> tuple[dict[int, dict], dict]:
    """Resolve per-seed instance data. Returns ({seed: instance_dict}, meta).

    source: file     -> load JSON, assert num_qubits / initial_state / array
                        lengths, and SEED COVERAGE (F2). Execution path stays
                        RNG-free.
    source: generate -> draw once per seed via a per-seed Generator (pcg64) or
                        legacy global np.random (legacy_npr, bit-exact
                        migration); materialize to the same schema.
    """
    source = disorder_spec.get("source", "file")

    if source == "file":
        with open(disorder_spec["file"]) as f:
            doc = json.load(f)
        meta = doc.get("_meta", {})
        instances = doc.get("instances", {})
        _assert_meta(meta, num_qubits, configured_initial_state)
        _assert_seed_coverage(instances, seed_values)          # F2
        resolved = {s: instances[str(s)] for s in seed_values}
        _assert_array_lengths(resolved, num_qubits)
        return resolved, meta

    if source == "generate":
        if sampler is None:
            raise ValueError(
                "disorder source 'generate' requires a sampler(rng, num_qubits) "
                "-> dict (e.g. the factory module's sample_disorder)"
            )
        generator = disorder_spec.get("generator", "pcg64")
        master_seed = disorder_spec.get("master_seed", 0)
        initial_state = disorder_spec.get("initial_state", configured_initial_state)
        resolved = {}
        for s in seed_values:
            rng = _spawn_rng(generator, master_seed, s)
            resolved[s] = sampler(rng, num_qubits)
        meta = {
            "generator": generator, "master_seed": master_seed,
            "num_qubits": num_qubits, "initial_state": initial_state,
        }
        return resolved, meta

    raise ValueError(f"disorder source must be 'file' or 'generate', got {source!r}")


def _spawn_rng(generator: str, master_seed: int, seed_index: int):
    """Per-seed RNG. pcg64 -> numpy.random.Generator (concurrency-safe, default).
    legacy_npr -> the legacy global np.random stream seeded for bit-exact
    migration of the banked run (serial, offline use only)."""
    if generator == "pcg64":
        child = np.random.SeedSequence(int(master_seed)).spawn(seed_index + 1)[seed_index]
        return np.random.Generator(np.random.PCG64(child))
    if generator == "legacy_npr":
        # Mirror floquet_runner.resolve_instance_seed -> uint32, then seed the
        # legacy global state. Caller's sampler uses np.random.* directly.
        child = np.random.SeedSequence(int(master_seed)).spawn(seed_index + 1)[seed_index]
        inst_seed = int(child.generate_state(1, dtype=np.uint32)[0])
        import random as _random
        _random.seed(inst_seed)
        np.random.seed(inst_seed)
        return np.random  # the module; sampler draws from the global stream
    raise ValueError(f"unknown disorder generator {generator!r}")


def _assert_meta(meta: dict, num_qubits: int, configured_initial_state: int | None) -> None:
    if "num_qubits" in meta and int(meta["num_qubits"]) != int(num_qubits):
        raise ValueError(
            f"disorder _meta.num_qubits={meta['num_qubits']} != "
            f"configured num_qubits={num_qubits}"
        )
    if (configured_initial_state is not None and "initial_state" in meta
            and int(meta["initial_state"]) != int(configured_initial_state)):
        raise ValueError(
            f"disorder _meta.initial_state={meta['initial_state']} != "
            f"configured initial_state={configured_initial_state}"
        )


def _assert_seed_coverage(instances: dict, seed_values: list[int]) -> None:
    missing = [s for s in seed_values if str(s) not in instances]
    if missing:
        raise ValueError(
            f"disorder JSON missing instances for seed(s) {missing}; "
            f"have {sorted(int(k) for k in instances)[:8]}{'...' if len(instances) > 8 else ''}"
        )


def _assert_array_lengths(resolved: dict[int, dict], num_qubits: int) -> None:
    for s, inst in resolved.items():
        for key, val in inst.items():
            if isinstance(val, list) and len(val) != num_qubits:
                raise ValueError(
                    f"disorder instance seed={s} field '{key}' has length "
                    f"{len(val)} != num_qubits {num_qubits}"
                )


# ═══════════════════════════════════════════════════════════════════════
# Per-task build seam: kwargs assembly + the default-ON cross-grid check
# ═══════════════════════════════════════════════════════════════════════

def assemble_build_kwargs(fixed: dict, instance: dict, grid_point: dict) -> dict:
    """Merge the three blocks for one (seed, grid-point). Key sets must be
    disjoint (validate_factory_signature enforces this up front)."""
    return {**fixed, **instance, **grid_point}


def cross_grid_identity_check(
    build_fn: Callable,
    *,
    fixed: dict,
    instance: dict,
    grid_points: list[dict],
    extract_disorder_params: Callable[[Any], Any],
    primary_axis: str | None = None,
) -> None:
    """§7.5.4 default-ON: within one seed, build the min and max grid points and
    assert their disorder-bearing structure is identical. Hard-fail on drift.

    `extract_disorder_params` pulls the disorder-derived structure from a built
    circuit (in the engine: the rz/rzz angle parameters). Kept as an argument so
    this logic is qiskit-independent and testable with a stub.
    """
    if len(grid_points) < 2:
        return
    if primary_axis is not None:
        ordered = sorted(grid_points, key=lambda p: p.get(primary_axis, 0))
    else:
        ordered = grid_points
    lo, hi = ordered[0], ordered[-1]
    c_lo = build_fn(**assemble_build_kwargs(fixed, instance, lo))
    c_hi = build_fn(**assemble_build_kwargs(fixed, instance, hi))
    p_lo = extract_disorder_params(c_lo)
    p_hi = extract_disorder_params(c_hi)
    if p_lo != p_hi:
        raise ValueError(
            "cross-grid disorder-identity check FAILED: disorder-bearing "
            f"structure differs between grid points {lo} and {hi} within a "
            "seed. The factory must be a pure function of (params, supplied "
            "disorder) and must not draw or reseed RNG at build time (§7.5.4)."
        )
