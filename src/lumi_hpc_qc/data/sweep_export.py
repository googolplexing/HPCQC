# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""E8 sweep export — HDF5 noise atlas → 61-column Parquet + summary CSV.

Reads the HDF5 file produced by E7's sweep engine and flattens every
result group into one row of the Parquet training table. This is the
Team Orange data interface — the boundary defined in CONTRACT_ORANGE_BLUE §5.1.

Architecture: metadata-scan export. Reads only HDF5 attributes and small
datasets (energy_trajectory, placement_qubits). Never reads large arrays
unless explicitly needed for derived features. This keeps the export
O(N) in attributes, not O(N × T) in trajectory length.

The 61-column schema is from RED-DIRECTIVE-E4-SCHEMA-v1.0 §4:
  - Identity & provenance (4)
  - Experiment configuration (11)
  - Hamiltonian properties (3)
  - Device & placement (7)
  - Calibration (8)
  - Noise & mitigation (4)
  - Circuit metrics (3)
  - Results (7)
  - Aggregated features (3)
  - Noise fingerprinting F1–F8 (11)

Validation targets:
  VE16: No raw histograms in Parquet (aggregated features only)
  VE17: Calibration columns present and populated
  VE23: Topology columns populated from topology library metadata

RED-SPEC-002 §9
RED-DIRECTIVE-E4-SCHEMA-v1.0 §4–§5
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from lumi_hpc_qc import __version__ as _pkg_version


# ═══════════════════════════════════════════════════════════════════════
# Parquet schema definition — 61 columns
# ═══════════════════════════════════════════════════════════════════════

def _build_parquet_schema():
    """Build the 61-column PyArrow schema.

    Lazy import to avoid pyarrow dependency at module load time
    (pyarrow is heavy and not needed for HDF5-only operations).
    """
    import pyarrow as pa

    return pa.schema([
        # ── Identity & Provenance (4) ──
        pa.field("experiment_id", pa.string()),
        pa.field("schema_version", pa.string()),
        pa.field("framework_version", pa.string()),
        pa.field("quality_gate_passed", pa.bool_()),

        # ── Experiment Configuration (11) ──
        pa.field("model", pa.string()),
        pa.field("ansatz", pa.string()),
        pa.field("optimizer", pa.string()),
        pa.field("gradient_method", pa.string()),
        pa.field("initializer", pa.string()),
        pa.field("num_qubits", pa.int32()),
        pa.field("ansatz_reps", pa.int32()),
        pa.field("num_parameters", pa.int32()),
        pa.field("optimizer_maxiter", pa.int32()),
        pa.field("shots", pa.int32()),
        pa.field("seed", pa.int32()),

        # ── Hamiltonian Properties (3) ──
        pa.field("spectral_gap", pa.float64()),             # nullable (>24q)
        pa.field("hamiltonian_locality", pa.int32()),
        pa.field("num_pauli_terms", pa.int32()),

        # ── Device & Placement (7) ──
        pa.field("device", pa.string()),
        pa.field("placement_qubits", pa.string()),
        pa.field("circuit_topology", pa.string()),
        pa.field("topology_equivalence_class", pa.string()),
        pa.field("placement_fidelity_score", pa.float64()),
        pa.field("submission_round", pa.int32()),
        pa.field("coupling_map_source", pa.string()),

        # ── Calibration (8) ──
        pa.field("calibration_source", pa.string()),
        pa.field("calibration_device", pa.string()),
        pa.field("calibration_date", pa.string()),
        pa.field("calibration_is_synthetic", pa.bool_()),
        pa.field("per_qubit_t1_us", pa.list_(pa.float64())),
        pa.field("per_qubit_t2_us", pa.list_(pa.float64())),
        pa.field("per_qubit_readout_fidelity", pa.list_(pa.float64())),
        pa.field("per_edge_cz_fidelity", pa.list_(pa.float64())),

        # ── Noise & Mitigation (4) ──
        pa.field("noise_environment", pa.string()),
        pa.field("noise_channels_active", pa.string()),
        pa.field("mitigation_readout", pa.bool_()),
        pa.field("mitigation_zne", pa.bool_()),

        # ── Circuit Metrics (3) ──
        pa.field("pre_transpilation_depth", pa.int32()),
        pa.field("post_transpilation_depth", pa.int32()),
        pa.field("swap_count", pa.int32()),

        # ── Results (7) ──
        pa.field("best_energy", pa.float64()),
        pa.field("exact_energy", pa.float64()),
        pa.field("relative_error", pa.float64()),
        pa.field("total_iterations", pa.int32()),
        pa.field("optimizer_converged", pa.bool_()),
        pa.field("wall_time_s", pa.float64()),
        pa.field("noiseless_tier", pa.int32()),             # nullable

        # ── Aggregated Features (3) ──
        pa.field("convergence_rate", pa.float64()),
        pa.field("energy_variance", pa.float64()),
        pa.field("final_gradient_norm", pa.float64()),      # nullable

        # ── Noise Fingerprinting (11) ──
        pa.field("measurement_entropy", pa.float64()),
        pa.field("dominant_bitstring_fraction", pa.float64()),
        pa.field("num_unique_bitstrings", pa.int32()),
        pa.field("bitstring_hamming_weight_mean", pa.float64()),       # F1
        pa.field("bitstring_hamming_weight_variance", pa.float64()),   # F2
        pa.field("z_group_expectation_mean", pa.float64()),            # F3
        pa.field("xz_expectation_ratio", pa.float64()),                # F4
        pa.field("effective_hilbert_dimension", pa.float64()),         # F5
        pa.field("kl_divergence_from_uniform", pa.float64()),          # F6
        pa.field("expectation_variance_across_groups", pa.float64()),  # F7
        pa.field("dominant_bitstring_hamming_weight", pa.int32()),     # F8
    ])


# ═══════════════════════════════════════════════════════════════════════
# Noise config metadata for export enrichment
# ═══════════════════════════════════════════════════════════════════════

def _noise_env_metadata(env_name: str) -> dict[str, Any]:
    """Return noise environment metadata for a given environment name.

    Enriches each row with fields derivable from the noise config name
    without needing the actual NoiseConfig object at export time.
    """
    from lumi_hpc_qc.sweep.noise_configs import (
        NOISE_ENV_BY_NAME, get_active_channels_string,
    )

    if env_name in NOISE_ENV_BY_NAME:
        nc = NOISE_ENV_BY_NAME[env_name]
        return {
            "noise_channels_active": get_active_channels_string(nc),
            "shots": nc.shots,
            "tier": nc.tier,
            "coupling_map_source": nc.coupling_map_source,
        }
    return {
        "noise_channels_active": "unknown",
        "shots": 0,
        "tier": "",
        "coupling_map_source": "unknown",
    }


def _noiseless_tier_int(tier: str) -> int | None:
    """Convert tier string to integer for Parquet."""
    return {"noiseless": 0, "A": 1, "B": 2, "full": 3}.get(tier)


# ═══════════════════════════════════════════════════════════════════════
# Derived feature computation
# ═══════════════════════════════════════════════════════════════════════

def _convergence_rate(trajectory: list[float] | np.ndarray) -> float | None:
    """Energy improvement per iteration over the last 20% of the trajectory.

    Returns the slope of a linear fit to the last 20% of energy values.
    Negative = still improving, ~0 = converged, positive = diverging.
    """
    if len(trajectory) < 5:
        return None
    n = len(trajectory)
    tail_start = max(1, int(n * 0.8))
    tail = np.array(trajectory[tail_start:], dtype=np.float64)
    if len(tail) < 2:
        return None
    x = np.arange(len(tail), dtype=np.float64)
    # Linear regression slope
    x_mean = x.mean()
    y_mean = tail.mean()
    denom = np.sum((x - x_mean) ** 2)
    if denom == 0:
        return 0.0
    return float(np.sum((x - x_mean) * (tail - y_mean)) / denom)


def _energy_variance(trajectory: list[float] | np.ndarray) -> float | None:
    """Variance of the last 10 energy evaluations (shot noise estimate)."""
    if len(trajectory) < 2:
        return None
    tail = np.array(trajectory[-10:], dtype=np.float64)
    return float(np.var(tail))


def _measurement_entropy(counts: dict[str, int] | None) -> float | None:
    """Shannon entropy of the count distribution."""
    if not counts:
        return None
    total = sum(counts.values())
    if total == 0:
        return None
    entropy = 0.0
    for c in counts.values():
        if c > 0:
            p = c / total
            entropy -= p * math.log2(p)
    return entropy


def _dominant_bitstring_fraction(counts: dict[str, int] | None) -> float | None:
    """Fraction of shots in the most-probable bitstring."""
    if not counts:
        return None
    total = sum(counts.values())
    if total == 0:
        return None
    return max(counts.values()) / total


def _num_unique_bitstrings(counts: dict[str, int] | None) -> int | None:
    """Number of distinct bitstrings observed."""
    if not counts:
        return None
    return len(counts)


def _bitstring_hamming_weight_mean(counts: dict[str, int] | None) -> float | None:
    """F1: Mean Hamming weight of observed bitstrings."""
    if not counts:
        return None
    total = sum(counts.values())
    if total == 0:
        return None
    hw_sum = sum(bs.count("1") * cnt for bs, cnt in counts.items())
    return hw_sum / total


def _bitstring_hamming_weight_variance(counts: dict[str, int] | None) -> float | None:
    """F2: Variance of Hamming weight across observed bitstrings."""
    if not counts:
        return None
    total = sum(counts.values())
    if total == 0:
        return None
    mean_hw = sum(bs.count("1") * cnt for bs, cnt in counts.items()) / total
    var_hw = sum(((bs.count("1") - mean_hw) ** 2) * cnt for bs, cnt in counts.items()) / total
    return var_hw


def _effective_hilbert_dimension(counts: dict[str, int] | None) -> float | None:
    """F5: Participation ratio 1/Σ(pᵢ²)."""
    if not counts:
        return None
    total = sum(counts.values())
    if total == 0:
        return None
    sum_p_sq = sum((c / total) ** 2 for c in counts.values())
    if sum_p_sq == 0:
        return None
    return 1.0 / sum_p_sq


def _kl_divergence_from_uniform(counts: dict[str, int] | None, num_qubits: int) -> float | None:
    """F6: KL divergence from uniform distribution. Convention: 0 × log(0) = 0."""
    if not counts or num_qubits == 0:
        return None
    total = sum(counts.values())
    if total == 0:
        return None
    n_states = 2 ** num_qubits
    q = 1.0 / n_states  # uniform probability
    kl = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            kl += p * math.log2(p / q)
    return kl


def _dominant_bitstring_hamming_weight(counts: dict[str, int] | None) -> int | None:
    """F8: Hamming weight of the most-probable bitstring."""
    if not counts:
        return None
    dominant = max(counts, key=counts.get)
    return dominant.count("1")


# ═══════════════════════════════════════════════════════════════════════
# Calibration extraction
# ═══════════════════════════════════════════════════════════════════════

def _extract_calibration_lists(
    per_qubit_cal: dict[str, dict[str, float]],
    qubit_names: list[str],
) -> dict[str, list[float]]:
    """Extract ordered per-qubit calibration arrays from the JSON dict.

    Returns lists aligned with placement qubit order.
    """
    t1_list = []
    t2_list = []
    ro_list = []

    for qname in qubit_names:
        qcal = per_qubit_cal.get(qname, {})
        t1_list.append(qcal.get("t1_us", 0.0))
        t2_list.append(qcal.get("t2_us", 0.0))
        ro_list.append(qcal.get("readout_fidelity", 0.0))

    return {
        "per_qubit_t1_us": t1_list,
        "per_qubit_t2_us": t2_list,
        "per_qubit_readout_fidelity": ro_list,
    }


# ═══════════════════════════════════════════════════════════════════════
# Main export function
# ═══════════════════════════════════════════════════════════════════════

def export_sweep_to_parquet(
    hdf5_path: str,
    output_path: str | None = None,
    *,
    exact_energies: dict[str, float] | None = None,
    include_csv: bool = True,
) -> dict[str, Any]:
    """Export an E7 sweep HDF5 file to 61-column Parquet.

    Walks every leaf group (those containing energy_trajectory),
    extracts attributes, computes derived features, and writes
    one Parquet row per result.

    Args:
        hdf5_path: Path to the sweep HDF5 file from E7.
        output_path: Output .parquet path. Default: same dir as HDF5.
        exact_energies: Optional dict of {model_name: exact_ground_energy}.
            Used to compute relative_error. If None, relative_error is null.
        include_csv: Also write a summary CSV (default True).

    Returns:
        Dict with export statistics:
          - parquet_path: str
          - csv_path: str | None
          - total_rows: int
          - columns: int
          - export_time_s: float
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    t_start = time.time()

    hdf5_p = Path(hdf5_path)
    if output_path is None:
        output_path = str(hdf5_p.with_suffix(".parquet"))
    csv_path = str(Path(output_path).with_suffix(".csv")) if include_csv else None

    schema = _build_parquet_schema()
    column_names = [f.name for f in schema]

    # Collect rows
    rows: list[dict[str, Any]] = []

    print(f"\n── E8 Sweep Export ──")
    print(f"  Source: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5:
        sweep_attrs = dict(h5.attrs)
        fw_version = str(sweep_attrs.get("framework_version", _pkg_version))

        # Walk leaf groups
        def _visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Group):
                return
            if "energy_trajectory" not in obj:
                return

            row = _extract_row(obj, name, fw_version, exact_energies)
            rows.append(row)

        h5.visititems(_visit)

    print(f"  Leaf groups found: {len(rows)}")

    if not rows:
        print("  WARNING: No result groups found in HDF5. Empty export.")
        return {
            "parquet_path": output_path,
            "csv_path": csv_path,
            "total_rows": 0,
            "columns": len(column_names),
            "export_time_s": time.time() - t_start,
        }

    # Build columnar arrays
    columns: dict[str, list] = {name: [] for name in column_names}
    for row in rows:
        for col_name in column_names:
            columns[col_name].append(row.get(col_name))

    # Build PyArrow table
    arrays = []
    for i, field in enumerate(schema):
        col_data = columns[field.name]
        try:
            arrays.append(pa.array(col_data, type=field.type))
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            # Fallback: let pyarrow infer, then cast
            try:
                arr = pa.array(col_data)
                arrays.append(arr.cast(field.type))
            except Exception:
                # Last resort: null array
                arrays.append(pa.nulls(len(col_data), type=field.type))

    table = pa.table({f.name: arrays[i] for i, f in enumerate(schema)},
                     schema=schema)

    # Write Parquet
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path, compression="snappy")
    print(f"  Parquet: {output_path} ({len(rows)} rows × {len(column_names)} columns)")

    # Write CSV summary
    if include_csv and csv_path:
        _write_summary_csv(rows, csv_path)
        print(f"  CSV: {csv_path}")

    elapsed = time.time() - t_start
    print(f"  Export time: {elapsed:.1f}s")

    return {
        "parquet_path": output_path,
        "csv_path": csv_path,
        "total_rows": len(rows),
        "columns": len(column_names),
        "export_time_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════════
# Row extraction from a single HDF5 group
# ═══════════════════════════════════════════════════════════════════════

def _extract_row(
    grp: h5py.Group,
    group_path: str,
    framework_version: str,
    exact_energies: dict[str, float] | None,
) -> dict[str, Any]:
    """Extract one Parquet row from an HDF5 leaf group.

    Reads only attributes and small datasets. Computes derived
    features from energy_trajectory when available.
    """
    attrs = dict(grp.attrs)

    # ── Parse JSON attributes ──
    circuit_metrics = {}
    cm_raw = attrs.get("circuit_metrics", "{}")
    if isinstance(cm_raw, str):
        try:
            circuit_metrics = json.loads(cm_raw)
        except json.JSONDecodeError:
            pass

    per_qubit_cal = {}
    pqc_raw = attrs.get("per_qubit_calibration", "{}")
    if isinstance(pqc_raw, str):
        try:
            per_qubit_cal = json.loads(pqc_raw)
        except json.JSONDecodeError:
            pass

    noise_fp = {}
    nfp_raw = attrs.get("noise_fingerprint", "{}")
    if isinstance(nfp_raw, str):
        try:
            noise_fp = json.loads(nfp_raw)
        except json.JSONDecodeError:
            pass

    edge_cz_fidelity = None
    ecf_raw = attrs.get("per_edge_cz_fidelity", None)
    if isinstance(ecf_raw, str):
        try:
            ecf_parsed = json.loads(ecf_raw)
            if isinstance(ecf_parsed, list):
                edge_cz_fidelity = ecf_parsed
        except json.JSONDecodeError:
            pass

    # ── Read datasets ──
    energy_traj = list(grp["energy_trajectory"][:]) if "energy_trajectory" in grp else []

    qubit_names = []
    if "placement_qubits" in grp:
        qubit_names = [
            q.decode("utf-8") if isinstance(q, bytes) else str(q)
            for q in grp["placement_qubits"][:]
        ]

    # ── Noise environment metadata ──
    noise_env = str(attrs.get("noise_config", ""))
    env_meta = _noise_env_metadata(noise_env)

    # ── Calibration extraction ──
    cal_lists = _extract_calibration_lists(per_qubit_cal, qubit_names)

    # ── Extract calibration date from cal_id ──
    cal_id = str(attrs.get("calibration_id", ""))
    cal_date = ""
    # Try to extract date from "cal_20260330" format
    if cal_id.startswith("cal_") and len(cal_id) >= 12:
        date_part = cal_id[4:]
        if date_part.isdigit() and len(date_part) == 8:
            cal_date = f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}"

    # ── Derived features ──
    best_energy = float(attrs.get("best_energy", 0.0))
    model_name = circuit_metrics.get("hamiltonian", "")
    num_qubits = int(circuit_metrics.get("num_qubits", 0))

    exact_energy = None
    relative_error = None
    if exact_energies and model_name in exact_energies:
        exact_energy = exact_energies[model_name]
        if exact_energy is not None and exact_energy != 0.0:
            relative_error = abs((best_energy - exact_energy) / abs(exact_energy))

    conv_rate = _convergence_rate(energy_traj)
    e_var = _energy_variance(energy_traj)

    # ── Build row ──
    row: dict[str, Any] = {
        # Identity & Provenance
        "experiment_id": str(attrs.get("experiment_id", "")),
        "schema_version": "2.0.0",
        "framework_version": str(attrs.get("framework_version", framework_version)),
        "quality_gate_passed": True,

        # Experiment Configuration
        "model": model_name,
        "ansatz": circuit_metrics.get("ansatz", "none"),
        "optimizer": circuit_metrics.get("optimizer", "none"),
        "gradient_method": circuit_metrics.get("gradient_method", "none"),
        "initializer": circuit_metrics.get("initializer", "none"),
        "num_qubits": num_qubits,
        "ansatz_reps": int(circuit_metrics.get("ansatz_reps", 0)),
        "num_parameters": int(circuit_metrics.get("num_parameters", 0)),
        "optimizer_maxiter": int(circuit_metrics.get("optimizer_maxiter", 0)),
        "shots": env_meta["shots"],
        "seed": int(attrs.get("seed", 0)),

        # Hamiltonian Properties
        "spectral_gap": circuit_metrics.get("spectral_gap"),
        "hamiltonian_locality": circuit_metrics.get("hamiltonian_locality"),
        "num_pauli_terms": circuit_metrics.get("num_pauli_terms"),

        # Device & Placement
        "device": str(attrs.get("device_id", "")),
        "placement_qubits": ",".join(qubit_names),
        "circuit_topology": circuit_metrics.get("topology_name", ""),
        "topology_equivalence_class": str(attrs.get("topology_hash", "")),
        "placement_fidelity_score": float(attrs.get("placement_score", 0.0)),
        "submission_round": 0,
        "coupling_map_source": env_meta["coupling_map_source"],

        # Calibration
        "calibration_source": cal_id,
        "calibration_device": str(attrs.get("device_id", "")),
        "calibration_date": cal_date,
        "calibration_is_synthetic": False,
        "per_qubit_t1_us": cal_lists["per_qubit_t1_us"] or None,
        "per_qubit_t2_us": cal_lists["per_qubit_t2_us"] or None,
        "per_qubit_readout_fidelity": cal_lists["per_qubit_readout_fidelity"] or None,
        "per_edge_cz_fidelity": edge_cz_fidelity,

        # Noise & Mitigation
        "noise_environment": noise_env,
        "noise_channels_active": env_meta["noise_channels_active"],
        "mitigation_readout": False,
        "mitigation_zne": False,

        # Circuit Metrics
        "pre_transpilation_depth": int(circuit_metrics.get("pre_transpilation_depth", 0)),
        "post_transpilation_depth": int(circuit_metrics.get("post_transpilation_depth", 0)),
        "swap_count": int(circuit_metrics.get("swap_count", 0)),

        # Results
        "best_energy": best_energy,
        "exact_energy": exact_energy,
        "relative_error": relative_error,
        "total_iterations": int(attrs.get("total_iterations", 0)),
        "optimizer_converged": bool(attrs.get("converged", False)),
        "wall_time_s": float(attrs.get("wall_time_seconds", 0.0)),
        "noiseless_tier": _noiseless_tier_int(env_meta.get("tier", "")),

        # Aggregated Features
        "convergence_rate": conv_rate,
        "energy_variance": e_var,
        "final_gradient_norm": noise_fp.get("final_gradient_norm"),

        # Noise Fingerprinting — from HDF5 noise_fingerprint attribute
        "measurement_entropy": noise_fp.get("measurement_entropy"),
        "dominant_bitstring_fraction": noise_fp.get("dominant_bitstring_fraction"),
        "num_unique_bitstrings": noise_fp.get("num_unique_bitstrings"),
        "bitstring_hamming_weight_mean": noise_fp.get("bitstring_hamming_weight_mean"),
        "bitstring_hamming_weight_variance": noise_fp.get("bitstring_hamming_weight_variance"),
        "z_group_expectation_mean": noise_fp.get("z_group_expectation_mean"),
        "xz_expectation_ratio": noise_fp.get("xz_expectation_ratio"),
        "effective_hilbert_dimension": noise_fp.get("effective_hilbert_dimension"),
        "kl_divergence_from_uniform": noise_fp.get("kl_divergence_from_uniform"),
        "expectation_variance_across_groups": noise_fp.get("expectation_variance_across_groups"),
        "dominant_bitstring_hamming_weight": noise_fp.get("dominant_bitstring_hamming_weight"),
    }

    return row


# ═══════════════════════════════════════════════════════════════════════
# Summary CSV
# ═══════════════════════════════════════════════════════════════════════

_CSV_COLUMNS = [
    "experiment_id", "model", "num_qubits", "seed", "device",
    "placement_qubits", "circuit_topology", "noise_environment",
    "calibration_source", "best_energy", "exact_energy", "relative_error",
    "total_iterations", "wall_time_s", "topology_equivalence_class",
    "placement_fidelity_score",
]


def _write_summary_csv(rows: list[dict[str, Any]], csv_path: str) -> None:
    """Write a summary CSV with key columns for quick inspection."""
    import csv as csv_mod

    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in _CSV_COLUMNS})


# ═══════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    """CLI: python -m lumi_hpc_qc.data.sweep_export sweep.h5 [--output sweep.parquet]"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m lumi_hpc_qc.data.sweep_export <sweep.h5> [--output <path.parquet>]")
        sys.exit(1)

    hdf5_path = sys.argv[1]
    output_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]

    result = export_sweep_to_parquet(hdf5_path, output_path)
    print(f"\nExported {result['total_rows']} rows to {result['parquet_path']}")


if __name__ == "__main__":
    main()
