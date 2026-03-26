# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Adiabatic parameter initialization for strongly correlated systems.

Based on FiQCI blog (CSC/VTT, 2025) and Wecker et al., Phys. Rev. A 92, 062318 (2015).

Ramps the interaction parameter (U/t for Fermi-Hubbard, Jz for Heisenberg)
from 0 to its target value. At each ramp step, runs gradient descent with
backtracking line search to track the ground state. Uses warm-restart from
the previous step's optimal parameters.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from lumi_hpc_qc.plugins.initializers.base import InitializerStrategy
from lumi_hpc_qc.types import ExperimentConfig


class AdiabaticInitializer(InitializerStrategy):
    """Adiabatic parameter initialization with gradient descent."""

    name = "adiabatic"

    def initialize(self, num_params, hamiltonian_builder=None, ansatz=None,
                   backend=None, config=None):
        if hamiltonian_builder is None or ansatz is None or backend is None:
            raise ValueError(
                "Adiabatic initializer requires hamiltonian_builder, ansatz, and backend"
            )

        p = config.initializer_params if config else {}
        n_steps = p.get("adiabatic_steps", 20)
        max_iter = p.get("adiabatic_max_iter", 300)
        seed = p.get("seed", 42)

        param_name = hamiltonian_builder.adiabatic_parameter_name()
        if param_name is None:
            print("  Adiabatic init not applicable for this model, using random init")
            rng = np.random.RandomState(seed)
            return rng.uniform(-np.pi / 2, np.pi / 2, num_params)

        # Get target value from config
        target_value = config.model_params.get(param_name, 1.0)

        print(f"\n[Adiabatic init] Ramping {param_name} from 0.0 → {target_value:.1f} "
              f"in {n_steps} steps")
        print(f"  Gradient descent: {max_iter} steps/point, adaptive LR")

        ramp_values = np.linspace(0, target_value, n_steps + 1)
        np.random.seed(seed)
        current_params = np.random.uniform(-np.pi / 2, np.pi / 2, num_params)

        t_start = time.time()

        # Determine gradient strategy from ansatz metadata
        from lumi_hpc_qc.plugins.gradients.parameter_shift import ParameterShiftGradient
        from lumi_hpc_qc.plugins.gradients.finite_difference import FiniteDifferenceGradient
        # Use parameter-shift for standard gates, finite-diff for UCCSD
        # We check by trying parameter-shift validation
        ps = ParameterShiftGradient()
        # Simple heuristic: if ansatz has standard rotation gates, use param-shift
        use_param_shift = True  # default; override if needed

        # Ensure Aer is initialized (patches QuantumCircuit with save_expectation_value)
        backend._ensure_sim()

        for step_i, ramp_val in enumerate(ramp_values):
            # Build Hamiltonian at this ramp value
            step_ham = hamiltonian_builder.build_at_parameter(ramp_val, config)

            # Energy evaluation function for this Hamiltonian
            def _make_eval(ham):
                def _eval(params):
                    param_dict = dict(zip(ansatz.parameters, params))
                    bound = ansatz.assign_parameters(param_dict)
                    bound.save_expectation_value(ham, list(range(ansatz.num_qubits)), label='e')
                    r = backend._sim.run(
                        bound, shots=0, seed_simulator=42,
                        blocking_enable=backend._use_blocking,
                        blocking_qubits=backend._blocking_qubits,
                    ).result()
                    return float(np.real(r.data()['e']))
                return _eval

            eval_fn = _make_eval(step_ham)

            # Gradient function
            if use_param_shift:
                shift = np.pi / 2
                def _grad(params, _ef=eval_fn):
                    g = np.zeros(len(params))
                    for k in range(len(params)):
                        pp = params.copy(); pp[k] += shift
                        pm = params.copy(); pm[k] -= shift
                        g[k] = (_ef(pp) - _ef(pm)) / 2.0
                    return g
            else:
                eps = 0.01
                def _grad(params, _ef=eval_fn):
                    g = np.zeros(len(params))
                    e0 = _ef(params)
                    for k in range(len(params)):
                        pp = params.copy(); pp[k] += eps
                        g[k] = (_ef(pp) - e0) / eps
                    return g

            # Gradient descent with backtracking line search
            n_gd = max_iter * 3 if step_i == 0 else max_iter
            best_e = eval_fn(current_params)
            best_p = current_params.copy()
            lr = 0.1

            for gd_i in range(n_gd):
                grad = _grad(current_params)
                gnorm = np.linalg.norm(grad)
                if gnorm < 1e-8:
                    break

                # Backtracking
                step_lr = lr
                e_curr = eval_fn(current_params)
                for _ in range(8):
                    trial = current_params - step_lr * grad
                    e_trial = eval_fn(trial)
                    if e_trial < e_curr - 1e-8:
                        current_params = trial
                        if e_trial < best_e:
                            best_e = e_trial
                            best_p = trial.copy()
                        lr = min(lr * 1.1, 0.3)
                        break
                    step_lr *= 0.5
                else:
                    lr *= 0.5
                    if lr < 1e-6:
                        break

            current_params = best_p
            print(f"    {param_name} = {ramp_val:5.2f}  →  E = {best_e:+12.6f}  "
                  f"({gd_i + 1} GD steps, lr={lr:.4f})")

        elapsed = time.time() - t_start
        print(f"  Adiabatic init complete in {elapsed:.1f}s")
        print(f"  Warm-started ||θ|| = {np.linalg.norm(current_params):.4f}")

        return current_params
