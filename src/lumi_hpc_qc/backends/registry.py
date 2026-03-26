# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Backend registry — maps string names to Backend subclasses.

Auto-discovers all Backend implementations in the backends/ package.
Used by controller.py to instantiate the right backend from config.

Usage:
    registry = BackendRegistry()
    registry.discover()
    backend = registry.get("aer_gpu", config)
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import TYPE_CHECKING

from lumi_hpc_qc.backends.base import Backend

if TYPE_CHECKING:
    from lumi_hpc_qc.types import ExperimentConfig


class BackendRegistry:
    """Factory that maps backend names to concrete implementations."""

    def __init__(self) -> None:
        self._backends: dict[str, type[Backend]] = {}

    def discover(self) -> None:
        """Scan the backends package and register all Backend subclasses.

        Walks every .py file in lumi_hpc_qc.backends/, imports it,
        and registers any class that subclasses Backend and has a
        non-empty `name` attribute.
        """
        import lumi_hpc_qc.backends as pkg

        for importer, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
            if modname in ("base", "registry", "__init__"):
                continue
            try:
                module = importlib.import_module(f"lumi_hpc_qc.backends.{modname}")
            except ImportError:
                continue
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, Backend)
                    and attr is not Backend
                    and getattr(attr, "name", "")
                ):
                    self._backends[attr.name] = attr

    def register(self, backend_cls: type[Backend]) -> None:
        """Manually register a backend class."""
        if not getattr(backend_cls, "name", ""):
            raise ValueError(f"Backend class {backend_cls} must have a non-empty 'name' attribute")
        self._backends[backend_cls.name] = backend_cls

    def get(self, name: str, config: ExperimentConfig | None = None) -> Backend:
        """Instantiate a backend by name.

        Args:
            name: Backend identifier (e.g., "aer_gpu", "iqm_q50")
            config: Experiment config passed to backend constructor

        Raises:
            KeyError: If no backend registered with this name
        """
        if name not in self._backends:
            available = ", ".join(sorted(self._backends.keys()))
            raise KeyError(
                f"Unknown backend '{name}'. Available: {available or '(none — call discover() first)'}"
            )
        return self._backends[name](config)

    def list_available(self) -> list[str]:
        """List all registered backend names."""
        return sorted(self._backends.keys())

    def validate_config(self, name: str, config: ExperimentConfig) -> list[str]:
        """Validate config against a specific backend without instantiating it."""
        backend = self.get(name, config)
        return backend.validate_config(config)
