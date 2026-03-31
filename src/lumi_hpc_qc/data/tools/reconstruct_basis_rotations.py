# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Reconstruct basis_rotations from pauli_group and grouping_algorithm.

For QWC (qubit-wise commuting) grouping, the basis rotation for each qubit
is deterministic: the non-identity Pauli at that position across all terms
in the group. This is the inverse of strip_basis_rotations.

Usage:
    from lumi_hpc_qc.data.tools import reconstruct_basis_rotations
    reconstruct_basis_rotations("experiments.h5")  # modifies in-place

    # Or from CLI:
    python -m lumi_hpc_qc.data.tools.reconstruct_basis_rotations experiments.h5
"""

from __future__ import annotations

import json


def _basis_from_pauli_group(pauli_labels: list[str]) -> dict[str, str]:
    """Derive per-qubit measurement basis from a list of QWC Pauli strings.

    For each qubit position, finds the non-identity Pauli operator.
    In a QWC group, all non-identity operators at the same position
    must agree (that's the definition of qubit-wise commutativity).

    Args:
        pauli_labels: e.g. ["ZZI", "ZIZ"]

    Returns:
        Dict mapping qubit index (as string) to Pauli char.
        e.g. {"0": "Z", "1": "Z", "2": "Z"}
    """
    basis = {}
    for label in pauli_labels:
        for pos, char in enumerate(reversed(label)):
            if char != "I":
                basis[str(pos)] = char
    return basis


def reconstruct_basis_rotations(
    hdf5_path: str, dry_run: bool = False
) -> dict:
    """Rebuild basis_rotations in every measurement_stats entry.

    Only operates on datasets with attrs["basis_rotations_stripped"] = True
    or entries that lack basis_rotations.

    Args:
        hdf5_path: Path to the HDF5 file (modified in-place unless dry_run).
        dry_run:   If True, report what would change without modifying.

    Returns:
        Dict with keys: experiments_processed, entries_reconstructed.
    """
    import h5py

    mode = "r" if dry_run else "r+"
    stats = {"experiments_processed": 0, "entries_reconstructed": 0}

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
            reconstructed = 0

            for line in lines:
                try:
                    entry = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    new_lines.append(line)
                    continue

                if "basis_rotations" not in entry and "pauli_group" in entry:
                    entry["basis_rotations"] = _basis_from_pauli_group(
                        entry["pauli_group"]
                    )
                    new_lines.append(json.dumps(entry, separators=(",", ":")))
                    reconstructed += 1
                else:
                    new_lines.append(line)

            if reconstructed > 0 and not dry_run:
                attrs_backup = {
                    "grouping_algorithm": ds.attrs.get("grouping_algorithm", "qwc"),
                    "interval": ds.attrs.get("interval", 10),
                    "num_entries": len(new_lines),
                }
                del exp_grp["measurement_stats"]
                dt = h5py.string_dtype(encoding="utf-8")
                new_ds = exp_grp.create_dataset(
                    "measurement_stats", data=new_lines, dtype=dt,
                    compression="gzip", compression_opts=4,
                )
                for k, v in attrs_backup.items():
                    new_ds.attrs[k] = v
                # Remove the stripped flag since we've reconstructed
                if "basis_rotations_stripped" in new_ds.attrs:
                    del new_ds.attrs["basis_rotations_stripped"]

            stats["entries_reconstructed"] += reconstructed
            stats["experiments_processed"] += 1

    return stats


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m lumi_hpc_qc.data.tools.reconstruct_basis_rotations "
              "<file.h5> [--dry-run]")
        sys.exit(1)

    path = sys.argv[1]
    dry = "--dry-run" in sys.argv

    result = reconstruct_basis_rotations(path, dry_run=dry)
    prefix = "[DRY RUN] " if dry else ""
    print(f"{prefix}Processed {result['experiments_processed']} experiments")
    print(f"{prefix}Reconstructed {result['entries_reconstructed']} entries")
