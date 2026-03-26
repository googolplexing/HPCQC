# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Workflow definitions — the orchestration brain.

A Workflow ties plugins + backends + data together into a complete
computational pipeline. Calls plugin interfaces, never concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lumi_hpc_qc.types import ExperimentConfig, ExperimentRecord, IterationRecord


class Workflow(ABC):
    """Base class for all computational workflows."""

    name: str = ""

    @abstractmethod
    def run(self, config: ExperimentConfig) -> ExperimentRecord:
        """Execute the full workflow from scratch."""

    @abstractmethod
    def resume(self, checkpoint_path: str, config: ExperimentConfig) -> ExperimentRecord:
        """Resume a previously interrupted workflow."""

    @abstractmethod
    def get_required_plugins(self) -> dict[str, str]:
        """Declare which plugin types this workflow needs."""


class VQEWorkflow(Workflow):
    """Variational Quantum Eigensolver — the core optimization loop."""

    name = "vqe"

    def run(self, config: ExperimentConfig) -> ExperimentRecord:
        ctx = self._setup(config)
        x0 = self._initialize_params(ctx)
        return self._optimize(ctx, x0, start_iteration=0)

    def resume(self, checkpoint_path: str, config: ExperimentConfig) -> ExperimentRecord:
        from lumi_hpc_qc.orchestration.checkpoint import CheckpointManager

        # Load checkpoint
        mgr = CheckpointManager(config.checkpoint.directory)
        state = mgr.load(checkpoint_path)
        meta = state.get("_checkpoint_meta", {})
        start_iter = meta.get("iteration", state.get("iteration", 0))

        print(f"\n  RESUMING from checkpoint: iteration {start_iter}")
        print(f"  Checkpoint: {checkpoint_path}")

        # Rebuild full pipeline
        ctx = self._setup(config)

        # Restore parameters from checkpoint
        x0 = state.get("params")
        if isinstance(x0, list):
            x0 = np.array(x0)
        best_e = state.get("best_energy", float('inf'))

        print(f"  Restored energy: {best_e:+14.8f}")
        print(f"  Restored ||θ||:  {float(np.linalg.norm(x0)):.4f}")

        # Adjust remaining iteration budget
        maxiter = config.optimizer_params.get("maxiter", 1000)
        remaining = max(maxiter - start_iter, 100)
        config.optimizer_params["maxiter"] = remaining
        print(f"  Remaining iterations: {remaining}")

        return self._optimize(ctx, x0, start_iteration=start_iter)

    def get_required_plugins(self) -> dict[str, str]:
        return {
            "hamiltonian": "required", "ansatz": "required",
            "optimizer": "required", "gradient": "optional",
            "initializer": "required", "error_mitigation": "optional",
        }

    # ── Internal: setup pipeline (shared by run + resume) ──

    def _setup(self, config: ExperimentConfig) -> dict:
        """Build the full VQE pipeline. Returns context dict used by optimize."""
        from lumi_hpc_qc.backends.registry import BackendRegistry
        from lumi_hpc_qc.data.experiment import ExperimentTracker
        from lumi_hpc_qc.data.provenance import ProvenanceCollector
        from lumi_hpc_qc.data.timing import TimingTracker
        from lumi_hpc_qc.orchestration.checkpoint import CheckpointManager
        from lumi_hpc_qc.plugins.registry import PluginRegistry

        timer = TimingTracker()

        plugins = PluginRegistry()
        plugins.discover()
        backends = BackendRegistry()
        backends.discover()

        provenance = ProvenanceCollector().capture()
        tracker = ExperimentTracker(config)
        tracker.start(provenance=provenance)
        checkpoint_mgr = CheckpointManager(config.checkpoint.directory)
        timer.mark("setup")

        print("=" * 70)
        print(f"  VQE Workflow — {config.model} / {config.ansatz}")
        print(f"  Experiment: {config.experiment_id}")
        print("=" * 70)

        # Step 1: Build Hamiltonian
        print(f"\n[Step 1] Building Hamiltonian: {config.model}")
        ham_builder = plugins.get_hamiltonian(config.model)
        hamiltonian, ham_meta = ham_builder.build(config)
        config.num_qubits = ham_meta.num_qubits
        print(f"  {ham_meta.description}")
        print(f"  Pauli terms: {ham_meta.pauli_term_count}")
        timer.mark("hamiltonian_build")

        # Step 2: Exact ground state
        print(f"\n[Step 2] Computing exact ground state energy...")
        exact_energy = ham_builder.exact_ground_energy(hamiltonian)
        if exact_energy is not None:
            print(f"  Exact ground state energy: {exact_energy:.8f}")
        else:
            print(f"  Exact diag infeasible for {ham_meta.num_qubits} qubits")
        timer.mark("exact_diag")

        # Step 3: Build ansatz (merge ham metadata for QAOA edge_list etc.)
        for key, val in ham_meta.physical_params.items():
            if key not in config.model_params:
                config.model_params[key] = val

        print(f"\n[Step 3] Building ansatz: {config.ansatz}")
        ansatz_builder = plugins.get_ansatz(config.ansatz)
        ansatz, ansatz_meta = ansatz_builder.build(ham_meta.num_qubits, config)
        print(f"  Parameters: {ansatz_meta.num_parameters}")
        print(f"  Gradient compatibility: {ansatz_meta.gradient_compatibility}")
        print(f"  Preferred initializer: {ansatz_meta.preferred_initializer}")
        timer.mark("ansatz_build")

        # Step 4: Initialize backend
        print(f"\n[Step 4] Initializing backend: {config.backend}")
        backend = backends.get(config.backend, config)
        if ansatz_meta.requires_decomposition:
            ansatz = backend.compile_circuit(ansatz)
        print(f"  Precision: {config.precision}, Qubits: {ham_meta.num_qubits}")

        # Ensure Aer imported (patches QuantumCircuit with save_expectation_value)
        backend._ensure_sim()
        timer.mark("backend_init")

        # Step 5: Resolve gradient strategy
        grad_strategy = None
        grad_name = config.gradient
        if grad_name not in ("none", ""):
            try:
                grad_strategy = plugins.get_gradient("parameter_shift")
                if not grad_strategy.validate_ansatz(ansatz_meta):
                    grad_strategy = plugins.get_gradient("finite_difference")
                    grad_name = "finite_difference"
            except KeyError:
                grad_strategy = plugins.get_gradient("finite_difference")
                grad_name = "finite_difference"

        return {
            "config": config, "plugins": plugins, "timer": timer,
            "tracker": tracker, "checkpoint_mgr": checkpoint_mgr,
            "ham_builder": ham_builder, "hamiltonian": hamiltonian,
            "ham_meta": ham_meta, "exact_energy": exact_energy,
            "ansatz": ansatz, "ansatz_meta": ansatz_meta,
            "backend": backend, "grad_strategy": grad_strategy,
            "grad_name": grad_name,
        }

    def _initialize_params(self, ctx: dict) -> np.ndarray:
        """Run the initializer plugin. Only called by run(), not resume()."""
        config = ctx["config"]
        plugins = ctx["plugins"]
        ansatz_meta = ctx["ansatz_meta"]
        timer = ctx["timer"]

        init_name = config.initializer
        if init_name == "auto":
            init_name = ansatz_meta.preferred_initializer

        print(f"\n[Step 5] Initializing parameters: {init_name}")
        initializer = plugins.get_initializer(init_name)
        x0 = initializer.initialize(
            num_params=ansatz_meta.num_parameters,
            hamiltonian_builder=ctx["ham_builder"],
            ansatz=ctx["ansatz"],
            backend=ctx["backend"],
            config=config,
        )
        print(f"  ||θ_0|| = {float(np.linalg.norm(x0)):.4f}")
        timer.mark("initialization")
        return x0

    def _optimize(self, ctx: dict, x0: np.ndarray, start_iteration: int = 0) -> ExperimentRecord:
        """Run the optimization loop. Shared by run() and resume()."""
        config = ctx["config"]
        plugins = ctx["plugins"]
        timer = ctx["timer"]
        tracker = ctx["tracker"]
        checkpoint_mgr = ctx["checkpoint_mgr"]
        hamiltonian = ctx["hamiltonian"]
        ham_meta = ctx["ham_meta"]
        exact_energy = ctx["exact_energy"]
        ansatz = ctx["ansatz"]
        backend = ctx["backend"]
        grad_strategy = ctx["grad_strategy"]
        grad_name = ctx["grad_name"]

        # Build energy evaluation function
        eval_count = [0]

        def eval_energy(params):
            param_dict = dict(zip(ansatz.parameters, params))
            bound = ansatz.assign_parameters(param_dict)
            bound.save_expectation_value(
                hamiltonian, list(range(ham_meta.num_qubits)), label='energy'
            )
            r = backend._sim.run(
                bound, shots=0, seed_simulator=42,
                blocking_enable=backend._use_blocking,
                blocking_qubits=backend._blocking_qubits,
            ).result()
            eval_count[0] += 1
            return float(np.real(r.data()['energy']))

        # Build gradient function
        grad_fn = None
        if grad_strategy is not None:
            def grad_fn(params):
                return grad_strategy.compute(eval_energy, params, backend)

        # Print optimizer info
        print(f"\n[Step 6] {'Starting' if start_iteration == 0 else 'Resuming'} VQE optimization")
        print(f"  Optimizer: {config.optimizer}")
        if grad_strategy:
            print(f"  Gradient: {grad_name} ({grad_strategy.circuits_per_gradient} circuits/gradient)")
        print(f"  Max iterations: {config.optimizer_params.get('maxiter', 200)}")
        print("-" * 70)

        # Check initial gradient and perturb if needed
        e0 = eval_energy(x0)
        print(f"\n  Initial energy: E(θ_0) = {e0:+14.8f}")
        if grad_fn is not None:
            g0 = grad_fn(x0)
            g0_norm = float(np.linalg.norm(g0))
            print(f"  Initial gradient norm:  ||∇E|| = {g0_norm:.8f}")
            if g0_norm < 1e-6:
                for scale in [0.1, 0.3, 0.5]:
                    print(f"  WARNING: gradient near-zero — perturbing with scale={scale}")
                    rng = np.random.RandomState(99)
                    x0 = x0 + rng.uniform(-scale, scale, len(x0))
                    e0 = eval_energy(x0)
                    g0 = grad_fn(x0)
                    g0_norm = float(np.linalg.norm(g0))
                    print(f"  Perturbed energy: {e0:+14.8f},  ||∇E|| = {g0_norm:.8f}")
                    if g0_norm > 1e-6:
                        break
        print()

        # Optimizer + callback
        optimizer = plugins.get_optimizer(config.optimizer)
        best_energy = [float('inf')]
        best_iteration = [0]
        best_params = [x0.copy()]

        def on_iteration(record: IterationRecord) -> None:
            record.iteration += start_iteration  # offset for resumed runs
            if record.energy < best_energy[0]:
                best_energy[0] = record.energy
                best_iteration[0] = record.iteration
                best_params[0] = record.parameters.copy() if record.parameters is not None else best_params[0]
                record.is_best = True

            tracker.log_iteration(record)

            marker = " ★ NEW BEST" if record.is_best else ""
            err_str = ""
            if exact_energy is not None:
                err_str = f"  |err|={abs(record.energy - exact_energy):.6f}"
            print(f"  Iter {record.iteration:4d}: E = {record.energy:+14.8f}"
                  f"  t = {record.elapsed_s:.2f}s{err_str}{marker}")

            # Checkpoint at interval
            if (config.checkpoint.enabled and
                    record.iteration % config.checkpoint.interval == 0):
                state = {
                    "params": record.parameters,
                    "best_energy": best_energy[0],
                    "best_params": best_params[0],
                    "iteration": record.iteration,
                }
                cp_path = checkpoint_mgr.save(state, record.iteration, config.experiment_id)
                print(f"    [checkpoint saved: iter {record.iteration}]")

        timer.mark("vqe_start")

        opt_result = optimizer.minimize(
            cost_fn=eval_energy, x0=x0, grad_fn=grad_fn,
            config=config, callback=on_iteration,
        )

        timer.mark("vqe_complete")

        # Finalize
        timing = timer.finish()
        total_iters = max(opt_result.nit, len(tracker._iterations))

        print("\n" + "-" * 70)
        print("  CONVERGENCE SUMMARY")
        print("-" * 70)
        print(f"  Total iterations        : {total_iters}")
        print(f"  Best energy             : {best_energy[0]:+14.8f}")
        if exact_energy is not None:
            abs_err = abs(best_energy[0] - exact_energy)
            if abs(exact_energy) > 1e-10:
                rel_err = abs_err / abs(exact_energy) * 100
            else:
                rel_err = abs_err * 100
                print(f"  NOTE: exact energy ≈ 0, reporting absolute error as %")
            print(f"  Exact ground state      : {exact_energy:+14.8f}")
            print(f"  Best absolute error     : {abs_err:14.8f}")
            print(f"  Best relative error     : {rel_err:13.6f}%")
        print(f"  Circuit evaluations     : {eval_count[0]}")
        print("-" * 70)
        print(f"\n{timing.to_human_readable()}")

        opt_result.nit = total_iters
        record = tracker.finalize(opt_result, timing, exact_energy)

        # Cleanup old checkpoints on success
        if config.checkpoint.enabled:
            deleted = checkpoint_mgr.cleanup(config.experiment_id, keep_latest=1)
            if deleted:
                print(f"  Cleaned up {deleted} old checkpoint(s)")

        print(f"\n  Results saved to: {config.output_dir}/{config.model}/")
        return record


class VQAWorkflow(Workflow):
    """Generic Variational Quantum Algorithm (non-eigensolver)."""
    name = "vqa"

    def run(self, config: ExperimentConfig) -> ExperimentRecord:
        raise NotImplementedError("VQAWorkflow — Phase 3")

    def resume(self, checkpoint_path: str, config: ExperimentConfig) -> ExperimentRecord:
        raise NotImplementedError("VQAWorkflow.resume() — Phase 3")

    def get_required_plugins(self) -> dict[str, str]:
        return {"ansatz": "required", "optimizer": "required"}


class CircuitSubmissionWorkflow(Workflow):
    """Direct circuit execution — no optimization loop."""
    name = "circuit_submission"

    def run(self, config: ExperimentConfig) -> ExperimentRecord:
        raise NotImplementedError("CircuitSubmissionWorkflow — Phase 3")

    def resume(self, checkpoint_path: str, config: ExperimentConfig) -> ExperimentRecord:
        raise NotImplementedError("CircuitSubmissionWorkflow.resume() — Phase 3")

    def get_required_plugins(self) -> dict[str, str]:
        return {"error_mitigation": "optional"}
