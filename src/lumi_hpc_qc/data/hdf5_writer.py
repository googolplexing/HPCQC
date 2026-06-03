# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""HDF5-first write-during-execution for Phase E sweep engine.

Replaces the JSON-first write pattern with HDF5 as the primary write
target during sweep execution. Eliminates the 84,000-file problem at
sweep scale by writing all results into a single HDF5 file.

Key features:
  - Atomic group writes: each result written as complete unit, flushed
  - WAL (write-ahead log): crash safety independent of Lustre SWMR
  - Context manager interface for sweep lifecycle
  - SWMR mode: monitoring can read while sweep writes (if Lustre supports)
  - Group naming: {device_prefix}-{qubit_names} per placement
  - Noiseless deduplication via HDF5 soft links

The WAL pattern:
  1. Each result serialized to memory buffer
  2. Buffer appended to flat WAL file (one fwrite, crash-safe)
  3. HDF5 group created and written
  4. On sweep completion, WAL replayed to verify consistency
  5. On crash recovery, WAL replayed to reconstruct lost HDF5 groups

Phase E — RED-DIRECTIVE-PHASE-E-ROADMAP-v1.0, System 5
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Generator

import h5py
import numpy as np


def _byo_wal_safe(result: dict[str, Any]) -> dict[str, Any]:
    """Coerce a BYO result dict to JSON-serializable for the WAL line
    (numpy ints/floats → python; arrays → lists). Pure; no I/O."""
    def _coerce(v):
        if isinstance(v, np.generic):
            return v.item()
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, (list, tuple)):
            return [_coerce(x) for x in v]
        return v
    return {k: _coerce(v) for k, v in result.items()}


@dataclass
class SweepResultEntry:
    """One result to write into the HDF5 sweep file.

    Represents a single (placement, calibration, noise_config, seed)
    execution result.
    """

    # Identification
    device_id: str
    device_prefix: str
    seed: int
    placement_qubits: list[str]     # physical qubit names
    calibration_id: str
    noise_config: str               # e.g., "noiseless", "noise_full"

    # Results
    energy_trajectory: list[float]
    best_energy: float
    total_iterations: int
    converged: bool

    # Optional arrays
    parameter_trajectory: list[list[float]] | None = None
    measurement_stats: list[str] | None = None

    # Metadata
    circuit_metrics: dict[str, Any] = field(default_factory=dict)
    per_qubit_calibration: dict[str, dict[str, float]] = field(
        default_factory=dict
    )
    placement_score: float = 0.0
    topology_hash: str = ""
    wall_time_seconds: float = 0.0
    framework_version: str = ""
    experiment_id: str = ""
    noise_fingerprint: dict[str, float | int | None] = field(
        default_factory=dict
    )
    per_edge_cz_fidelity: list[float] | None = None
    exact_ground_energy: float | None = None
    model_params: dict[str, float] = field(default_factory=dict)
    calibration_set_id: str | None = None              # v1.4.0 — VTT QX calibration UUID
    packing_co_placements: int = 1                     # v1.4.0 — tasks in batch
    packing_qubit_utilization: float = 0.0             # v1.4.0 — batch utilization
    packing_algorithm: str = "none"                    # v1.4.0 — "dsatur"|"global_pool"|"none"

    @property
    def group_path(self) -> str:
        """HDF5 group path for this result — delegates to the single source of
        truth (``battery_paths.battery_group_path``) so the on-disk path, the
        merge extractor's parse, and the option-(i) expected-group inventory
        cannot drift (RED-RULING-PATCH43-VERIFY-AND-INVENTORY-DESIGN Q2).

        Format: devices/{device_prefix}/seeds/seed_{seed:04d}/
                placements/{device_prefix}-{qubit_names}/
                calibrations/{calibration_id}/{noise_config}/[params_{hash}]
        (the params suffix appears only in LHS mode; grid mode → unchanged).
        """
        from lumi_hpc_qc.sweep.battery_paths import battery_group_path
        return battery_group_path(
            self.device_prefix, self.seed, self.placement_qubits,
            self.calibration_id, self.noise_config, self.model_params,
        )

    def to_wal_dict(self) -> dict[str, Any]:
        """Serialize to a WAL-safe dict (JSON-serializable)."""
        d: dict[str, Any] = {
            "group_path": self.group_path,
            "device_id": self.device_id,
            "device_prefix": self.device_prefix,
            "seed": self.seed,
            "placement_qubits": self.placement_qubits,
            "calibration_id": self.calibration_id,
            "noise_config": self.noise_config,
            "best_energy": self.best_energy,
            "total_iterations": self.total_iterations,
            "converged": self.converged,
            "energy_trajectory": self.energy_trajectory,
            "circuit_metrics": self.circuit_metrics,
            "per_qubit_calibration": self.per_qubit_calibration,
            "placement_score": self.placement_score,
            "topology_hash": self.topology_hash,
            "wall_time_seconds": self.wall_time_seconds,
            "framework_version": self.framework_version,
            "experiment_id": self.experiment_id,
        }
        if self.parameter_trajectory is not None:
            d["parameter_trajectory"] = self.parameter_trajectory
        if self.measurement_stats is not None:
            d["measurement_stats"] = self.measurement_stats
        if self.noise_fingerprint:
            d["noise_fingerprint"] = self.noise_fingerprint
        if self.per_edge_cz_fidelity is not None:
            d["per_edge_cz_fidelity"] = self.per_edge_cz_fidelity
        if self.exact_ground_energy is not None:
            d["exact_ground_energy"] = self.exact_ground_energy
        if self.model_params:
            d["model_params"] = self.model_params
        if self.calibration_set_id is not None:
            d["calibration_set_id"] = self.calibration_set_id
        d["packing_co_placements"] = self.packing_co_placements
        d["packing_qubit_utilization"] = self.packing_qubit_utilization
        d["packing_algorithm"] = self.packing_algorithm
        return d


class SweepHDF5Writer:
    """HDF5-first writer for sweep execution.

    Usage:
        with SweepHDF5Writer("sweep.h5", sweep_attrs={...}) as writer:
            for result in sweep_results:
                writer.write(result)

        # On crash recovery:
        writer = SweepHDF5Writer("sweep.h5")
        writer.recover_from_wal()
    """

    def __init__(
        self,
        hdf5_path: str,
        sweep_attrs: dict[str, Any] | None = None,
        enable_swmr: bool = False,
        wal_path: str | None = None,
        debug_json: bool = False,
        debug_json_dir: str | None = None,
        byo_collision_stems: set[str] | None = None,
    ) -> None:
        self._hdf5_path = Path(hdf5_path)
        self._wal_path = Path(wal_path or str(hdf5_path) + ".wal")
        self._sweep_attrs = sweep_attrs or {}
        self._enable_swmr = enable_swmr
        self._debug_json = debug_json
        self._debug_json_dir = Path(debug_json_dir) if debug_json_dir else None
        # BYO-FAMILY-COLLISION fix (b1): script stems hosting >1 circuit family
        # in this run (computed once at run level). For these, the default
        # family's leaf is disambiguated by circuit_function via the shared
        # byo_observable_subpath seam — in lockstep with the .dat aggregator.
        # Empty/None => no collision => legacy "" layout (byte-identical).
        self._byo_collision_stems = byo_collision_stems or set()
        self._h5file: h5py.File | None = None
        self._wal_file = None
        self._write_count = 0
        self._opened = False

    def open(self) -> None:
        """Open HDF5 and WAL files for writing."""
        if self._opened:
            return

        # Create parent directories
        self._hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        self._wal_path.parent.mkdir(parents=True, exist_ok=True)

        if self._debug_json and self._debug_json_dir:
            self._debug_json_dir.mkdir(parents=True, exist_ok=True)

        # Open HDF5
        mode = "a" if self._hdf5_path.exists() else "w"
        self._h5file = h5py.File(
            str(self._hdf5_path), mode,
            libver="latest",
        )

        # Enable SWMR if requested and file is new
        if self._enable_swmr and mode == "w":
            try:
                self._h5file.swmr_mode = True
            except Exception as e:
                print(f"  SWMR mode failed (expected on Lustre): {e}")
                print("  Continuing without SWMR — WAL provides crash safety")

        # Write sweep-level attributes
        for key, value in self._sweep_attrs.items():
            try:
                self._h5file.attrs[key] = value
            except TypeError:
                self._h5file.attrs[key] = str(value)

        # Open WAL (append mode)
        self._wal_file = open(self._wal_path, "a")

        self._opened = True

    def close(self) -> None:
        """Close HDF5 and WAL files. Verify consistency."""
        if self._h5file is not None:
            self._h5file.flush()
            self._h5file.close()
            self._h5file = None

        if self._wal_file is not None:
            self._wal_file.close()
            self._wal_file = None

        self._opened = False

    def __enter__(self) -> SweepHDF5Writer:
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def write(self, entry: SweepResultEntry) -> None:
        """Write one result to WAL + HDF5.

        Steps:
          1. Serialize to WAL (crash-safe append)
          2. Write HDF5 group with datasets and attributes
          3. Flush HDF5
          4. Optionally write debug JSON
        """
        if not self._opened:
            raise RuntimeError("Writer not opened. Use 'with' or call open().")

        # Step 1: WAL write (crash-safe — one line append + fsync)
        wal_line = json.dumps(entry.to_wal_dict()) + "\n"
        self._wal_file.write(wal_line)
        self._wal_file.flush()
        os.fsync(self._wal_file.fileno())

        # Step 2: HDF5 group write
        self._write_hdf5_group(entry)

        # Step 3: Flush HDF5
        self._h5file.flush()

        # Step 4: Optional debug JSON
        if self._debug_json and self._debug_json_dir:
            safe_name = entry.group_path.replace("/", "_")
            debug_path = self._debug_json_dir / f"{safe_name}.json"
            with open(debug_path, "w") as f:
                json.dump(entry.to_wal_dict(), f, indent=2)

        self._write_count += 1

    def _write_hdf5_group(self, entry: SweepResultEntry) -> None:
        """Write a single result as an HDF5 group."""
        group_path = entry.group_path

        # Create group (and all parent groups)
        if group_path in self._h5file:
            # Overwrite: delete and recreate
            del self._h5file[group_path]

        grp = self._h5file.create_group(group_path)

        # Datasets
        grp.create_dataset(
            "energy_trajectory",
            data=np.array(entry.energy_trajectory, dtype=np.float64),
        )

        if entry.parameter_trajectory is not None:
            grp.create_dataset(
                "parameter_trajectory",
                data=np.array(entry.parameter_trajectory, dtype=np.float64),
            )

        if entry.measurement_stats is not None:
            # Flat string array — each element is one JSON line
            dt = h5py.string_dtype(encoding="utf-8")
            grp.create_dataset(
                "measurement_stats",
                data=np.array(entry.measurement_stats, dtype=object),
                dtype=dt,
            )

        # Scalar attributes
        grp.attrs["best_energy"] = entry.best_energy
        grp.attrs["total_iterations"] = entry.total_iterations
        grp.attrs["converged"] = entry.converged
        grp.attrs["wall_time_seconds"] = entry.wall_time_seconds
        grp.attrs["experiment_id"] = entry.experiment_id
        grp.attrs["framework_version"] = entry.framework_version
        grp.attrs["placement_score"] = entry.placement_score
        grp.attrs["topology_hash"] = entry.topology_hash
        grp.attrs["noise_config"] = entry.noise_config
        grp.attrs["seed"] = entry.seed
        grp.attrs["device_id"] = entry.device_id
        grp.attrs["calibration_id"] = entry.calibration_id

        # Placement qubit names
        dt_str = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset(
            "placement_qubits",
            data=np.array(entry.placement_qubits, dtype=object),
            dtype=dt_str,
        )

        # Per-qubit calibration as JSON attribute
        if entry.per_qubit_calibration:
            grp.attrs["per_qubit_calibration"] = json.dumps(
                entry.per_qubit_calibration
            )

        # Circuit metrics as JSON attribute
        if entry.circuit_metrics:
            grp.attrs["circuit_metrics"] = json.dumps(
                entry.circuit_metrics
            )

        # Noise fingerprinting features as JSON attribute
        if entry.noise_fingerprint:
            grp.attrs["noise_fingerprint"] = json.dumps(
                entry.noise_fingerprint
            )

        # Per-edge CZ fidelity as JSON attribute
        if entry.per_edge_cz_fidelity is not None:
            grp.attrs["per_edge_cz_fidelity"] = json.dumps(
                entry.per_edge_cz_fidelity
            )

        # Exact ground energy (v1.2.0 Item D — RED-SPEC-003)
        # float64 for ≤24 qubits, omitted for >24 qubits
        if entry.exact_ground_energy is not None:
            grp.attrs["exact_ground_energy"] = entry.exact_ground_energy

        # Model parameters as JSON attribute (v1.2.0 Item C — RED-SPEC-003)
        # Stores LHS-sampled or user-specified Hamiltonian params
        if entry.model_params:
            grp.attrs["model_params"] = json.dumps(entry.model_params)

        # v1.4.0 — packing provenance + calibration UUID
        if entry.calibration_set_id is not None:
            grp.attrs["calibration_set_id"] = entry.calibration_set_id
        grp.attrs["packing_co_placements"] = entry.packing_co_placements
        grp.attrs["packing_qubit_utilization"] = entry.packing_qubit_utilization
        grp.attrs["packing_algorithm"] = entry.packing_algorithm

    # ── BYO counts results (SPEC-002 §7.5 / D3.4c, Option A) ──
    # The BYO counts→autocorrelator observable is a per-kick VECTOR, not an
    # energy trajectory, so it gets its own group tree under /byo rather than
    # being forced through the energy-shaped SweepResultEntry. The 71-col
    # physics-Parquet extension is a separate, Red-reviewed step.
    def write_byo_result(self, result: dict[str, Any]) -> None:
        """Write one BYO (seed × placement × env) autocorrelator series.

        Group path:
          /byo/{script_stem}/seeds/seed_{seed:04d}/
              placements/{phys_qubits_joined}/{env}

        Datasets:  autocorrelator (float64[N_kicks]), num_kicks (int[N_kicks]),
                   physical_qubit_set (utf-8[n_qubits]).
        Attrs:     noise_source, noise_placement_independent, seed,
                   seed_simulator, master_seed, shots, placement_id,
                   optimization_level, calibration_set_id (NF4).

        ``result`` is one dict from SweepEngine._byo_results_last.
        """
        if not self._opened:
            raise RuntimeError("Writer not opened. Use 'with' or call open().")

        script_stem = Path(result["script"]).stem if result.get("script") else "byo"
        phys = "-".join(str(q) for q in result["physical_qubit_set"])
        # D7 increment 2: observable level appended via the shared helper —
        # "" for the synthesized "default" (LEGACY path, byte-identical pre-D7,
        # so the W1.6 gate / banked references are untouched) or "/<name>" for a
        # declared family. Local import: sweep/__init__ pulls sweep_engine, which
        # imports this module, so a module-level sweep import here would cycle.
        from lumi_hpc_qc.sweep.byo_observable import (
            DEFAULT_OBSERVABLE_NAME,
            byo_observable_subpath,
        )
        observable = result.get("observable", DEFAULT_OBSERVABLE_NAME)
        # BYO-FAMILY-COLLISION fix (b1): disambiguate a default-family leaf by
        # circuit_function iff this script stem hosts >1 family in the run
        # (the run-level set passed at construction). Same seam + same flag the
        # .dat aggregator uses, so the HDF5 and .dat layouts cannot drift.
        circuit_function = result.get("circuit_function")
        disambiguate = script_stem in self._byo_collision_stems
        # No leading slash — matches SweepResultEntry.group_path ("devices/...")
        # and the names h5py.visititems reports (root-relative). h5py.create_group
        # places it at /byo/... regardless, and f["/byo/..."] still resolves it.
        group_path = (
            f"byo/{script_stem}/seeds/seed_{int(result['seed']):04d}/"
            f"placements/{phys}/{result['env']}"
            f"{byo_observable_subpath(observable, circuit_function, disambiguate)}"
        )

        # WAL append (crash-safe), tagged so recovery can distinguish BYO rows.
        # The computed group_path is stored in the WAL line so it is symmetric
        # with the energy path (SweepResultEntry.to_wal_dict carries group_path):
        # both verify_consistency and recover_from_wal key off group_path, so a
        # BYO line WITHOUT it would be silently dropped by recovery and would
        # inject "" into verify_consistency's wal_paths (spurious inconsistency).
        wal_line = json.dumps(
            {"_kind": "byo", "group_path": group_path, **_byo_wal_safe(result)}
        ) + "\n"
        self._wal_file.write(wal_line)
        self._wal_file.flush()
        os.fsync(self._wal_file.fileno())

        self._write_byo_hdf5_group(result, group_path)
        self._h5file.flush()
        self._write_count += 1

    def _write_byo_hdf5_group(
        self, result: dict[str, Any], group_path: str
    ) -> None:
        """Create the BYO HDF5 group (datasets + attrs) at ``group_path``.

        Pure HDF5 write — no WAL append, no flush — so it can be reused by both
        ``write_byo_result`` (live path) and ``recover_from_wal`` (replay path,
        which must NOT re-append to the WAL). ``result`` may be either a live
        ``_byo_results_last`` dict or a WAL-replayed dict (same keys).
        """
        if group_path in self._h5file:
            del self._h5file[group_path]
        grp = self._h5file.create_group(group_path)

        grp.create_dataset(
            "autocorrelator",
            data=np.array(result["autocorrelator"], dtype=np.float64),
        )
        grp.create_dataset(
            "num_kicks",
            data=np.array(result["num_kicks"], dtype=np.int64),
        )
        dt_str = h5py.string_dtype(encoding="utf-8")
        grp.create_dataset(
            "physical_qubit_set",
            data=np.array([str(q) for q in result["physical_qubit_set"]], dtype=object),
            dtype=dt_str,
        )

        grp.attrs["noise_source"] = result["noise_source"]
        grp.attrs["noise_placement_independent"] = bool(
            result["noise_placement_independent"]
        )
        grp.attrs["seed"] = int(result["seed"])
        if result.get("seed_simulator") is not None:
            grp.attrs["seed_simulator"] = int(result["seed_simulator"])
        # RED-RESP-D3.4C §3: master_seed is the parent knob seed_simulator is
        # derived from (seed_simulator = resolve_instance_seed(master_seed,
        # seed)). Store it whenever known (0 is a valid value -> `is not None`)
        # so a stored result is traceable to the run that produced it. Absent
        # only when the disorder carried no master_seed (entropy / unrepeatable).
        if result.get("master_seed") is not None:
            grp.attrs["master_seed"] = int(result["master_seed"])
        grp.attrs["shots"] = int(result["shots"])
        if result.get("placement_id") is not None:
            grp.attrs["placement_id"] = int(result["placement_id"])
        # NF4 (W1.4): persist the W1 provenance attrs that the result dict +
        # WAL already carry but the writer previously dropped. optimization_level
        # is physics-affecting under noise (CFG-2); calibration_set_id ties the
        # record to its calibration. .get() so WAL-replay (which feeds the same
        # _write_byo_hdf5_group via _byo_wal_safe, a whole-dict coercion) and the
        # live path behave identically. Criterion 6 (RED-RESP-W1.3-VERIFY NF4).
        if result.get("optimization_level") is not None:
            grp.attrs["optimization_level"] = int(result["optimization_level"])
        if result.get("calibration_set_id") is not None:
            grp.attrs["calibration_set_id"] = str(result["calibration_set_id"])

    def create_soft_link(
        self, source_path: str, target_path: str
    ) -> None:
        """Create an HDF5 soft link for noiseless deduplication.

        When the same placement topology produces identical noiseless
        results across calibrations, the second calibration links to
        the first instead of storing duplicate data.
        """
        if not self._opened:
            raise RuntimeError("Writer not opened.")
        self._h5file[target_path] = h5py.SoftLink(source_path)

    def recover_from_wal(self) -> int:
        """Replay WAL to recover from a crash.

        Reads the WAL file, checks which entries are missing from HDF5,
        and writes the missing ones.

        Returns:
            Number of entries recovered.
        """
        if not self._wal_path.exists():
            print("  No WAL file found — nothing to recover")
            return 0

        # Open HDF5 for writing
        self._h5file = h5py.File(str(self._hdf5_path), "a", libver="latest")
        recovered = 0

        with open(self._wal_path) as wal:
            for line_num, line in enumerate(wal, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    wal_dict = json.loads(line)
                except json.JSONDecodeError:
                    print(f"  WAL line {line_num}: corrupt, skipping")
                    continue

                group_path = wal_dict.get("group_path", "")
                if not group_path:
                    continue

                # Check if this group exists in HDF5
                if group_path in self._h5file:
                    continue  # Already written

                # Reconstruct entry and write. BYO rows (D3.4c) are not
                # energy-shaped, so they cannot go through _wal_dict_to_entry /
                # _write_hdf5_group — replay them via the BYO group writer.
                if wal_dict.get("_kind") == "byo":
                    self._write_byo_hdf5_group(wal_dict, group_path)
                else:
                    entry = self._wal_dict_to_entry(wal_dict)
                    self._write_hdf5_group(entry)
                recovered += 1

        self._h5file.flush()
        self._h5file.close()
        self._h5file = None

        if recovered > 0:
            print(f"  WAL recovery: {recovered} entries restored")
        else:
            print("  WAL recovery: all entries already in HDF5")

        return recovered

    def _wal_dict_to_entry(self, d: dict[str, Any]) -> SweepResultEntry:
        """Reconstruct a SweepResultEntry from a WAL dict."""
        return SweepResultEntry(
            device_id=d["device_id"],
            device_prefix=d["device_prefix"],
            seed=d["seed"],
            placement_qubits=d["placement_qubits"],
            calibration_id=d["calibration_id"],
            noise_config=d["noise_config"],
            energy_trajectory=d["energy_trajectory"],
            best_energy=d["best_energy"],
            total_iterations=d["total_iterations"],
            converged=d["converged"],
            parameter_trajectory=d.get("parameter_trajectory"),
            measurement_stats=d.get("measurement_stats"),
            circuit_metrics=d.get("circuit_metrics", {}),
            per_qubit_calibration=d.get("per_qubit_calibration", {}),
            placement_score=d.get("placement_score", 0.0),
            topology_hash=d.get("topology_hash", ""),
            wall_time_seconds=d.get("wall_time_seconds", 0.0),
            framework_version=d.get("framework_version", ""),
            experiment_id=d.get("experiment_id", ""),
            noise_fingerprint=d.get("noise_fingerprint", {}),
            per_edge_cz_fidelity=d.get("per_edge_cz_fidelity"),
            exact_ground_energy=d.get("exact_ground_energy"),
            model_params=d.get("model_params", {}),
            calibration_set_id=d.get("calibration_set_id"),
            packing_co_placements=d.get("packing_co_placements", 1),
            packing_qubit_utilization=d.get("packing_qubit_utilization", 0.0),
            packing_algorithm=d.get("packing_algorithm", "none"),
        )

    def verify_consistency(self) -> dict[str, Any]:
        """Compare WAL entries against HDF5 contents.

        Returns a report dict with counts and any discrepancies.
        """
        if not self._wal_path.exists():
            return {"wal_entries": 0, "hdf5_groups": 0, "missing": 0}

        wal_paths = set()
        with open(self._wal_path) as wal:
            for line in wal:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    wal_paths.add(d.get("group_path", ""))
                except json.JSONDecodeError:
                    pass

        h5 = h5py.File(str(self._hdf5_path), "r")
        hdf5_paths = set()

        def collect_groups(name, obj):
            # Energy leaf groups carry "energy_trajectory"; BYO leaf groups
            # (D3.4c, /byo tree) carry "autocorrelator". Count both so a BYO or
            # mixed run does not spuriously report WAL inconsistency.
            if isinstance(obj, h5py.Group) and (
                "energy_trajectory" in obj or "autocorrelator" in obj
            ):
                hdf5_paths.add(name)

        h5.visititems(collect_groups)
        h5.close()

        missing = wal_paths - hdf5_paths
        extra = hdf5_paths - wal_paths

        return {
            "wal_entries": len(wal_paths),
            "hdf5_groups": len(hdf5_paths),
            "missing_from_hdf5": len(missing),
            "extra_in_hdf5": len(extra),
            "consistent": len(missing) == 0,
        }

    @property
    def write_count(self) -> int:
        """Number of results written in this session."""
        return self._write_count
