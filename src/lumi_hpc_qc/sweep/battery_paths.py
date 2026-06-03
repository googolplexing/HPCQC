# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Single source of truth for the battery (hamiltonian) HDF5 group path and its
group key (RED-RULING-PATCH43-VERIFY-AND-INVENTORY-DESIGN Q2).

The wholly-absent-group inventory (option (i)) compares the groups a merge
actually unioned against the groups the engine *intended*. That comparison is
sound only if the inventory's keys are byte-identical to what the reducer's
``extract`` yields from the written paths. To make that true BY CONSTRUCTION
rather than by coincidence, exactly one place builds the battery group path and
exactly one place parses it back into a key:

  - ``battery_group_path(...)``  — the path BUILDER. ``SweepResultEntry.group_path``
    delegates to it (so the writer's on-disk path is this string), and the
    inventory generator calls it directly (no fragile throwaway entry).
  - ``group_key_from_path(group_path)`` — the path PARSER. ``BatteryReducer.extract``
    parses each unit's GROUP path (``dirname`` of the energy_trajectory dataset)
    through it, and the inventory keys its built paths through the SAME parser.

Then ``group_key_from_path(battery_group_path(fields)) == extract's key`` for the
same logical unit, because both sides share the identical builder + parser. The
``params_{md5(...)}`` suffix is the most likely drift point, so it lives here once.

stdlib only (hashlib) — importable and unit-testable offline (no h5py/numpy).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence


def battery_group_path(
    device_prefix: str,
    seed: int,
    placement_qubits: Sequence[str],
    calibration_id: str,
    noise_config: str,
    model_params: Mapping[str, float] | None,
) -> str:
    """The HDF5 GROUP path for one battery unit (no trailing dataset). Format:

        devices/{device_prefix}/seeds/seed_{seed:04d}/
        placements/{device_prefix}-{qubit_names joined by _}/
        calibrations/{calibration_id}/{noise_config}[/params_{hash}]

    The ``params_{hash}`` suffix appears only when ``model_params`` is non-empty
    (LHS mode); it prevents path collisions when LHS samples share
    (device, seed, placement, calibration, noise_config) but differ in
    Hamiltonian parameters. Grid mode → empty model_params → no suffix.

    This is the byte-for-byte path the writer puts on disk (the
    ``SweepResultEntry.group_path`` property delegates here), and the path the
    inventory generator enumerates. Do NOT reformat any segment elsewhere.
    """
    qubit_str = "_".join(placement_qubits)
    base = (
        f"devices/{device_prefix}/"
        f"seeds/seed_{seed:04d}/"
        f"placements/{device_prefix}-{qubit_str}/"
        f"calibrations/{calibration_id}/"
        f"{noise_config}"
    )
    if model_params:
        params_str = ",".join(
            f"{k}={v:.8f}" for k, v in sorted(model_params.items())
        )
        params_hash = hashlib.md5(params_str.encode()).hexdigest()[:8]
        return f"{base}/params_{params_hash}"
    return base


def group_key_from_path(group_path: str) -> tuple[str, str, str, str, str] | None:
    """Parse a battery GROUP path into its group key
    ``(device_prefix, placement, calibration_id, noise_config, params_tail)``.

    ``group_path`` is the path of the per-unit leaf GROUP (i.e. the ``dirname`` of
    the ``energy_trajectory`` dataset, or equivalently the output of
    ``battery_group_path``) — NOT the dataset path. ``params_tail`` is everything
    after ``noise_config`` (the ``params_{hash}`` segment in LHS mode, else "").
    The seed lives in the path but is the INSTANCE axis, not part of the group
    key, so it is deliberately dropped here. Returns ``None`` for a path that is
    not a battery unit group (so a caller can skip foreign subtrees).
    """
    parts = group_path.split("/")
    try:
        di = parts.index("devices")
        pi = parts.index("placements")
        ci = parts.index("calibrations")
    except ValueError:
        return None
    device_prefix = parts[di + 1]
    placement = parts[pi + 1]
    cal_id = parts[ci + 1]
    noise_config = parts[ci + 2]
    params_tail = "/".join(parts[ci + 3:])  # params_HASH in LHS mode, else ""
    return (device_prefix, placement, cal_id, noise_config, params_tail)


# ── option-(i) inventory schema (one place; engine writes, merge CLI reads) ──

_INVENTORY_SCHEMA = "campaign_expected/v1"


def inventory_to_json(groups, reducer_name: str = "BatteryReducer") -> dict:
    """Serialize the expected-group set to the campaign_expected.json shape.
    Tuples become lists (JSON has no tuple); ``groups`` is sorted for a stable,
    diff-friendly file. The engine writes this once per shard run."""
    return {
        "schema": _INVENTORY_SCHEMA,
        "reducer": reducer_name,
        "groups": sorted(list(g) for g in groups),
    }


def inventory_from_json(d: dict) -> set[tuple]:
    """Rehydrate the expected-group SET from campaign_expected.json (lists ->
    tuples), fail-loud on an unexpected schema. The merge CLI uses this; the
    assertion is then a pure set comparison (offline-checkable)."""
    schema = d.get("schema")
    if schema != _INVENTORY_SCHEMA:
        raise ValueError(
            f"campaign_expected.json: unexpected schema {schema!r} "
            f"(expected {_INVENTORY_SCHEMA!r})."
        )
    return {tuple(g) for g in d.get("groups", [])}
