# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Strip basis_rotations from HDF5 measurement_stats to reduce size.

The basis_rotations field is fully deterministic from the pauli_group and
grouping_algorithm. Stripping it saves ~30% of the measurement_stats
dataset size. Use reconstruct_basis_rotations.py to rebuild it.

Usage:
    from lumi_hpc_qc.data.tools import strip_basis_rotations
    strip_basis_rotations("experiments.h5")  # modifies in-place

    # Or from CLI:
    python -m lumi_hpc_qc.data.tools.strip_basis_rotations experiments.h5
"""

from __future__ import annotations

import json


def strip_basis_rotations(hdf5_path: str, dry_run: bool = False) -> dict:
    """Remove basis_rotations from every measurement_stats entry in an HDF5 file.

    Args:
        hdf5_path: Path to the HDF5 file (modified in-place unless dry_run).
        dry_run:   If True, report what would change without modifying.

    Returns:
        Dict with keys: experiments_processed, entries_stripped, bytes_saved_estimate.
    """
    import h5py

    mode = "r" if dry_run else "r+"
    stats = {"experiments_processed": 0, "entries_stripped": 0, "bytes_saved_estimate": 0}

    with h5py.File(hdf5_path, mode) as hf:
        if "experiments" not in hf:
            return stats

        for exp_name in hf["experiments"]:
            exp_grp = hf[f"experiments/{exp_name}"]
            if "measurement_stats" not in exp_grp:
                continue

            ds = exp_grp["measurement_stats"]
            lines = list(ds[:])
            new_lines = []
            stripped = 0

            for line in lines:
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    new_lines.append(line)
                    continue

                if "basis_rotations" in entry:
                    old_len = len(line)
                    del entry["basis_rotations"]
                    new_line = json.dumps(entry, separators=(",", ":"))
                    stats["bytes_saved_estimate"] += old_len - len(new_line)
                    new_lines.append(new_line)
                    stripped += 1
                else:
                    new_lines.append(line)

            if stripped > 0 and not dry_run:
                # Replace dataset with stripped version
                del exp_grp["measurement_stats"]
                import h5py as h5
                dt = h5.string_dtype(encoding="utf-8")
                new_ds = exp_grp.create_dataset(
                    "measurement_stats", data=new_lines, dtype=dt,
                    compression="gzip", compression_opts=4,
                )
                # Preserve attributes
                new_ds.attrs["grouping_algorithm"] = ds.attrs.get(
                    "grouping_algorithm", "qwc"
                )
                new_ds.attrs["interval"] = ds.attrs.get("interval", 10)
                new_ds.attrs["num_entries"] = len(new_lines)
                new_ds.attrs["basis_rotations_stripped"] = True

            stats["entries_stripped"] += stripped
            stats["experiments_processed"] += 1

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m lumi_hpc_qc.data.tools.strip_basis_rotations <file.h5> [--dry-run]")
        sys.exit(1)

    path = sys.argv[1]
    dry = "--dry-run" in sys.argv

    result = strip_basis_rotations(path, dry_run=dry)
    prefix = "[DRY RUN] " if dry else ""
    print(f"{prefix}Processed {result['experiments_processed']} experiments")
    print(f"{prefix}Stripped {result['entries_stripped']} entries")
    print(f"{prefix}Estimated savings: {result['bytes_saved_estimate']} bytes")
