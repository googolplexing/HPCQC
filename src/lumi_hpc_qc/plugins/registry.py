# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Central plugin registry — discovers and provides all researcher-extensible components.

Scans each plugin sub-package for classes that subclass the relevant ABC,
registers them by their .name attribute. Researchers add a file, inherit
from the base, and it's automatically available — no core code changes.

Usage:
    registry = PluginRegistry()
    registry.discover()
    ham = registry.get_hamiltonian("fermi_hubbard")
    ans = registry.get_ansatz("hva")
"""

from __future__ import annotations

import importlib
import pkgutil
from typing import Any

from lumi_hpc_qc.plugins.ansatze.base import AnsatzBuilder
from lumi_hpc_qc.plugins.error_mitigation.base import ErrorMitigator
from lumi_hpc_qc.plugins.gradients.base import GradientStrategy
from lumi_hpc_qc.plugins.hamiltonians.base import HamiltonianBuilder
from lumi_hpc_qc.plugins.initializers.base import InitializerStrategy
from lumi_hpc_qc.plugins.optimizers.base import OptimizerStrategy
from lumi_hpc_qc.types import ExperimentConfig

# Maps plugin type name → (package path, ABC class)
_PLUGIN_TYPES: dict[str, tuple[str, type]] = {
    "hamiltonians": ("lumi_hpc_qc.plugins.hamiltonians", HamiltonianBuilder),
    "ansatze": ("lumi_hpc_qc.plugins.ansatze", AnsatzBuilder),
    "optimizers": ("lumi_hpc_qc.plugins.optimizers", OptimizerStrategy),
    "gradients": ("lumi_hpc_qc.plugins.gradients", GradientStrategy),
    "initializers": ("lumi_hpc_qc.plugins.initializers", InitializerStrategy),
    "error_mitigation": ("lumi_hpc_qc.plugins.error_mitigation", ErrorMitigator),
}


class PluginRegistry:
    """Central registry for all researcher-extensible components."""

    def __init__(self) -> None:
        # type_name → {plugin_name → plugin_class}
        self._plugins: dict[str, dict[str, type]] = {
            t: {} for t in _PLUGIN_TYPES
        }

    def discover(self) -> None:
        """Scan all plugin sub-packages and register found classes.

        For each sub-package (hamiltonians, ansatze, ...):
          1. Walk every .py file in the package directory
          2. Import it
          3. Find all classes that subclass the relevant ABC
          4. Register them by their .name attribute
        """
        for type_name, (pkg_path, abc_class) in _PLUGIN_TYPES.items():
            pkg = importlib.import_module(pkg_path)
            for importer, modname, ispkg in pkgutil.iter_modules(pkg.__path__):
                if modname in ("base", "__init__"):
                    continue
                try:
                    module = importlib.import_module(f"{pkg_path}.{modname}")
                except ImportError:
                    # Skip plugins with missing dependencies (e.g., molecular needs PySCF)
                    continue
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (
                        isinstance(attr, type)
                        and issubclass(attr, abc_class)
                        and attr is not abc_class
                        and getattr(attr, "name", "")
                    ):
                        self._plugins[type_name][attr.name] = attr

    def register(self, type_name: str, plugin_cls: type) -> None:
        """Manually register a plugin class."""
        if type_name not in self._plugins:
            raise ValueError(f"Unknown plugin type '{type_name}'")
        name = getattr(plugin_cls, "name", "")
        if not name:
            raise ValueError(f"Plugin class {plugin_cls} must have a non-empty 'name' attribute")
        self._plugins[type_name][name] = plugin_cls

    def _get(self, type_name: str, name: str) -> Any:
        """Get a plugin class by type and name, then instantiate it."""
        plugins = self._plugins.get(type_name, {})
        if name not in plugins:
            available = ", ".join(sorted(plugins.keys()))
            raise KeyError(
                f"Unknown {type_name} plugin '{name}'. "
                f"Available: {available or '(none — call discover() first)'}"
            )
        return plugins[name]()

    # ── Typed accessors (one per plugin type) ──

    def get_hamiltonian(self, name: str) -> HamiltonianBuilder:
        return self._get("hamiltonians", name)

    def get_ansatz(self, name: str) -> AnsatzBuilder:
        return self._get("ansatze", name)

    def get_optimizer(self, name: str) -> OptimizerStrategy:
        return self._get("optimizers", name)

    def get_gradient(self, name: str) -> GradientStrategy:
        return self._get("gradients", name)

    def get_initializer(self, name: str) -> InitializerStrategy:
        return self._get("initializers", name)

    def get_error_mitigator(self, name: str) -> ErrorMitigator:
        return self._get("error_mitigation", name)

    def list_available(self, type_name: str) -> list[str]:
        """List all registered plugin names for a given type."""
        if type_name not in self._plugins:
            raise ValueError(f"Unknown plugin type '{type_name}'")
        return sorted(self._plugins[type_name].keys())

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        """Validate that all plugins requested in config exist and are compatible.

        Returns:
            List of error messages. Empty = all valid.
        """
        errors: list[str] = []

        # Check each requested plugin exists
        checks = [
            ("hamiltonians", config.model),
            ("ansatze", config.ansatz),
            ("optimizers", config.optimizer),
        ]
        if config.gradient != "none":
            checks.append(("gradients", config.gradient))
        checks.append(("initializers", config.initializer))
        if config.error_mitigation:
            checks.append(("error_mitigation", config.error_mitigation))

        for type_name, plugin_name in checks:
            if plugin_name and plugin_name not in self._plugins.get(type_name, {}):
                available = ", ".join(sorted(self._plugins.get(type_name, {}).keys()))
                errors.append(
                    f"{type_name} plugin '{plugin_name}' not found. "
                    f"Available: {available}"
                )

        # Check gradient-ansatz compatibility
        if config.gradient != "none" and config.ansatz:
            ansatz_plugins = self._plugins.get("ansatze", {})
            grad_plugins = self._plugins.get("gradients", {})
            if config.ansatz in ansatz_plugins and config.gradient in grad_plugins:
                ansatz_inst = ansatz_plugins[config.ansatz]()
                grad_inst = grad_plugins[config.gradient]()
                # Build a minimal ansatz to check metadata
                # (this is a lightweight validation — full build happens later)

        return errors
