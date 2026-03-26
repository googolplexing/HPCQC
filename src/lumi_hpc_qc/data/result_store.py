# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Result store — serialization/deserialization of experiment results.

Handles numpy arrays, complex numbers, and other non-JSON types.
Defines the canonical result schema. Pure I/O utility — no business logic.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: Any) -> Any:
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype)}
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, complex):
            return {"__complex__": True, "real": obj.real, "imag": obj.imag}
        return super().default(obj)


def numpy_decoder_hook(obj: dict) -> Any:
    """JSON decoder hook that restores numpy types."""
    if "__ndarray__" in obj:
        return np.array(obj["data"], dtype=obj.get("dtype", "float64"))
    if "__complex__" in obj:
        return complex(obj["real"], obj["imag"])
    return obj


def save_json(data: Any, path: str | Path) -> None:
    """Write data to JSON file with numpy support."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)


def load_json(path: str | Path) -> Any:
    """Read JSON file with numpy type restoration."""
    with open(path) as f:
        return json.load(f, object_hook=numpy_decoder_hook)
