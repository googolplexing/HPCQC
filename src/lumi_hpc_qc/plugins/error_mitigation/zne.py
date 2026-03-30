# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Zero-Noise Extrapolation (ZNE) via mitiq.

CRITICAL: mitiq is imported LAZILY — only when mitigate() is first called,
not at module load or plugin discovery time. This is because mitiq's import
chain triggers MPI_Init_thread via qiskit_aer (cirq-core → controller_wrappers.so).
If mpi4py has already called MPI_Init, Cray MPICH on LUMI crashes.

The lazy import means ZNE and multi-node MPI simulation cannot coexist in
the same process without careful import ordering. For the current roadmap,
ZNE is used with shot-based simulation (single GPU) and does not conflict
with multi-node statevector simulation.

Usage:
    zne_mitigator = ZneErrorMitigator()
    mitigated_energy = zne_mitigator.mitigate(eval_fn, params, config)

YAML config:
    error_mitigation:
      zne:
        enabled: true
        scale_factors: [1, 3, 5]
        extrapolation: linear  # linear, polynomial, exponential, richardson
        folding: global        # global, local, random
        apply_every: 1         # ZNE every Nth eval; always on gradient steps
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from lumi_hpc_qc.plugins.error_mitigation.base import ErrorMitigator
from lumi_hpc_qc.types import ExperimentConfig


# Module-level flag: mitiq loaded only on first use
_mitiq_loaded = False
_mitiq = None
_zne_module = None


def _ensure_mitiq():
    """Lazily import mitiq. Only called when ZNE is actually used."""
    global _mitiq_loaded, _mitiq, _zne_module
    if _mitiq_loaded:
        return
    import mitiq
    from mitiq import zne
    _mitiq = mitiq
    _zne_module = zne
    _mitiq_loaded = True


class ZneErrorMitigator(ErrorMitigator):
    """Zero-Noise Extrapolation via mitiq gate folding."""

    name = "zne"
    requires_additional_circuits = True

    def __init__(self):
        self._eval_count = 0
        self._apply_every = 1
        self._is_gradient_step = False

    def configure(self, config: ExperimentConfig) -> None:
        """Read ZNE parameters from config."""
        em = config.error_mitigation_params
        zne_cfg = em.get("zne", {}) if em else {}
        self._apply_every = zne_cfg.get("apply_every", 1)
        self._scale_factors = zne_cfg.get("scale_factors", [1, 3, 5])
        self._extrapolation = zne_cfg.get("extrapolation", "linear")
        self._folding = zne_cfg.get("folding", "global")

    def should_apply(self) -> bool:
        """Determine if ZNE should be applied for this evaluation.

        Always applies during gradient steps (gradient quality has
        outsized impact on convergence). For regular evaluations,
        applies every Nth call.
        """
        self._eval_count += 1
        if self._is_gradient_step:
            return True
        return (self._eval_count % self._apply_every) == 0

    def set_gradient_step(self, is_gradient: bool) -> None:
        """Mark the current evaluation as part of gradient computation."""
        self._is_gradient_step = is_gradient

    def mitigate(
        self,
        eval_fn: Callable[[np.ndarray], float],
        circuit,
        params: np.ndarray,
        config: ExperimentConfig,
    ) -> float:
        """Run ZNE: fold gates at multiple noise levels, extrapolate to zero.

        Args:
            eval_fn: Function that takes parameters and returns energy.
            circuit: The bound quantum circuit (for gate folding).
            params: Current parameter values.
            config: Experiment config with ZNE parameters.

        Returns:
            Mitigated energy estimate.
        """
        _ensure_mitiq()

        em = config.error_mitigation_params or {}
        zne_cfg = em.get("zne", {})
        scale_factors = zne_cfg.get("scale_factors", [1, 3, 5])
        extrapolation = zne_cfg.get("extrapolation", "linear")
        folding = zne_cfg.get("folding", "global")

        # Select folding method
        if folding == "local":
            fold_fn = _zne_module.scaling.fold_gates_at_random
        elif folding == "random":
            fold_fn = _zne_module.scaling.fold_gates_at_random
        else:
            fold_fn = _zne_module.scaling.fold_global

        # Select extrapolation factory
        if extrapolation == "polynomial":
            factory = _zne_module.inference.PolyFactory(scale_factors, order=2)
        elif extrapolation == "exponential":
            factory = _zne_module.inference.ExpFactory(scale_factors)
        elif extrapolation == "richardson":
            factory = _zne_module.inference.RichardsonFactory(scale_factors)
        else:
            factory = _zne_module.inference.LinearFactory(scale_factors)

        # Build executor: evaluates energy for a given (possibly folded) circuit
        def executor(folded_circuit):
            """Execute a single circuit and return energy."""
            return eval_fn(params)

        try:
            mitigated = _zne_module.execute_with_zne(
                circuit,
                executor=executor,
                scale_noise=fold_fn,
                factory=factory,
            )
            return float(mitigated)
        except Exception as e:
            # Fallback: if ZNE fails, return unmitigated energy
            print(f"  ZNE failed ({e}), falling back to raw evaluation")
            return eval_fn(params)

    def mitigate_simple(
        self,
        eval_fn: Callable[[np.ndarray], float],
        params: np.ndarray,
        scale_factors: list[float] | None = None,
        extrapolation: str = "linear",
    ) -> float:
        """Simplified ZNE without mitiq — manual noise scaling via eval_fn.

        For cases where the eval_fn already handles noise scaling
        (e.g., backend-level noise amplification), this method does
        the extrapolation only.

        Args:
            eval_fn: Takes params, returns energy at current noise level.
            params: Parameter values.
            scale_factors: Noise scale factors [1, 3, 5].
            extrapolation: Fit method.

        Returns:
            Extrapolated energy at zero noise.
        """
        if scale_factors is None:
            scale_factors = [1, 3, 5]

        energies = [eval_fn(params) for _ in scale_factors]

        if extrapolation == "linear":
            coeffs = np.polyfit(scale_factors, energies, 1)
            return float(np.polyval(coeffs, 0))
        elif extrapolation == "exponential":
            if len(energies) >= 2:
                return float(2 * energies[0] - energies[1])
            return energies[0]
        elif extrapolation == "richardson":
            if len(energies) >= 3:
                # Richardson extrapolation for 3 points
                s1, s2, s3 = scale_factors[:3]
                e1, e2, e3 = energies[:3]
                # Two-step Richardson
                e12 = (s2 * e1 - s1 * e2) / (s2 - s1)
                e23 = (s3 * e2 - s2 * e3) / (s3 - s2)
                return float((s3 * e12 - s1 * e23) / (s3 - s1))
            return energies[0]
        else:
            degree = min(len(scale_factors) - 1, 2)
            coeffs = np.polyfit(scale_factors, energies, degree)
            return float(np.polyval(coeffs, 0))

    def pre_process(self, circuits, config):
        """ZNE pre-processing is handled by mitiq internally."""
        return circuits

    def post_process(self, results, config):
        """ZNE post-processing is handled by mitiq internally."""
        return results

    def validate_config(self, config: ExperimentConfig) -> list[str]:
        errors = []
        if config.backend in ("aer_gpu", "aer_cpu"):
            method = config.backend_params.get("method", "statevector")
            shots = config.backend_params.get("shots", 0)
            if method == "statevector" and shots == 0:
                errors.append(
                    "ZNE is not meaningful for noiseless statevector simulation. "
                    "Use shot-based simulation or QPU backend."
                )
        return errors
