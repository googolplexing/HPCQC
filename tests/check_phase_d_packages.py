# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Quick check that all Phase D export dependencies are present in the container."""
import sys

missing = []

try:
    import jsonschema
    print(f"jsonschema: {jsonschema.__version__}")
except ImportError:
    print("jsonschema: MISSING")
    missing.append("jsonschema")

try:
    import pyarrow
    print(f"pyarrow:    {pyarrow.__version__}")
except ImportError:
    print("pyarrow:    MISSING")
    missing.append("pyarrow")

try:
    import h5py
    print(f"h5py:       {h5py.__version__}")
except ImportError:
    print("h5py:       MISSING")
    missing.append("h5py")

if missing:
    print(f"\nMISSING packages: {', '.join(missing)}")
    sys.exit(1)
else:
    print("\nAll Phase D export dependencies present.")
