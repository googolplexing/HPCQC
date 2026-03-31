# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Quick check that all Phase D export dependencies are present in the container."""
import sys
import importlib.metadata

missing = []

for package in ("jsonschema", "pyarrow", "h5py"):
    try:
        __import__(package)
        version = importlib.metadata.version(package)
        print(f"{package:<12} {version}")
    except (ImportError, importlib.metadata.PackageNotFoundError):
        print(f"{package:<12} MISSING")
        missing.append(package)

if missing:
    print(f"\nMISSING packages: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll Phase D export dependencies present.")
