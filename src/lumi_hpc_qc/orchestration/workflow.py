# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Workflow definitions — the orchestration brain.

Fixes applied:
  F2: eval_energy and eval_energy_batch route through Backend.run_circuits()
      instead of bypassing to backend._sim.run(). This means:
      - Shot-based config (noisy/QPU) actually uses shots + readout noise
      - QPU backend works (no more crash on missing _sim)
      - Dependency inversion principle restored
  C6: seed_simulator incremented per evaluation for realistic shot noise
  Q3: CircuitSubmissionWorkflow.get_required_plugins() lists all used plugins
  Q4: VQAWorkflow dead class removed

A Workflow ties plugins + backends + data together into a complete
computational pipeline. Calls plugin interfaces, never concrete implementations.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from lumi_hpc_qc.types import (
    CircuitJob,
    ExperimentConfig,
    ExperimentRecord,
    IterationRecord,
)


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

        mgr = CheckpointManager(config.checkpoint.directory)
        state = mgr.load(checkpoint_path)
        meta = state.get("_checkpoint_meta", {})
        start_iter = meta.get("iteration", state.get("iteration", 0))

        print(f"\n  Resuming from checkpoint: iteration {start_iter}")
        print(f"  Checkpoint: {checkpoint_path}")

        ctx = self._setup(config)
        x0 = np.array(state["params"])
        print(f"  Restored parameters: ||θ|| = {float(np.linalg.norm(x0)):.4f}")

        if "best_energy" in state:
            print(f"  Previous best energy: {state['best_energy']:+14.8f}")

        return self._optimize(ctx, x0, start_iteration=start_iter)

    def get_required_plugins(self) -> dict[str, str]:
        return {
            "hamiltonian": "required", "ansatz": "required",
            "optimizer": "required", "gradient": "optional",
            "initializer": "required", "error_mitigation": "optional",
        }

    def _setup(self, config: ExperimentConfig) -> dict:
        """Build all components. Shared by run() and resume()."""
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

        print("=" * 70)
        print(f"  VQE — {config.model} / {config.ansatz} / {config.optimizer}")
        print(f"  Experiment: {config.experiment_id}")
        print("=" * 70)

        # Step 1: Build Hamiltonian
        print(f"\n[Step 1] Building Hamiltonian: {config.model}")
        ham_builder = plugins.get_hamiltonian(config.model)
        hamiltonian, ham_meta = ham_builder.build(config)
        config.num_qubits = ham_meta.num_qubits
        print(f"  Qubits: {ham_meta.num_qubits}, Pauli terms: {ham_meta.num_pauli_terms}")
        timer.mark("hamiltonian_build")

        # Step 2: Exact reference
        print(f"\n[Step 2] Computing exact ground state energy...")
        exact_energy = ham_builder.exact_ground_energy(hamiltonian)
        if exact_energy is not None:
            print(f"  Exact ground state energy: {exact_energy:.8f}")
        else:
            print(f"  System too large for exact diagonalization")

        # Phase D: enrich HamiltonianMetadata with locality and spectral_gap
        # Done here (not in plugins) so all 5 plugins benefit without modification.
        from lumi_hpc_qc.types import compute_spectral_gap, compute_hamiltonian_locality
        ham_meta.hamiltonian_locality = compute_hamiltonian_locality(hamiltonian)
        ham_meta.spectral_gap = compute_spectral_gap(hamiltonian, ham_meta.num_qubits)
        if ham_meta.spectral_gap is not None:
            print(f"  Spectral gap: {ham_meta.spectral_gap:.6f}")

        timer.mark("exact_diag")

        # Step 3: Build ansatz
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

        # Step 4a: Topology transpilation (Phase B)
        # Transpile ONCE in setup — not per eval_energy call.
        # seed_transpiler=42 ensures deterministic SWAP placement.
        circuit_metrics = None
        if hasattr(backend, '_coupling_map') and backend._coupling_map is not None:
            from qiskit import transpile
            from lumi_hpc_qc.types import CircuitMetrics

            # Record pre-transpilation metrics
            pre_depth = ansatz.depth()
            pre_gates = ansatz.size()
            pre_cx = ansatz.count_ops().get('cx', 0) + ansatz.count_ops().get('cz', 0)

            # Phase D: store on ansatz_meta before transpile overwrites the circuit
            ansatz_meta.pre_transpilation_depth = pre_depth
            ansatz_meta.pre_transpilation_cx_count = pre_cx

            backend._ensure_sim()  # ensure coupling map is loaded

            ansatz = transpile(
                ansatz,
                coupling_map=backend._coupling_map,
                optimization_level=2,
                seed_transpiler=42,
            )

            # Record post-transpilation metrics
            post_depth = ansatz.depth()
            post_gates = ansatz.size()
            post_cx = ansatz.count_ops().get('cx', 0) + ansatz.count_ops().get('cz', 0)
            swap_count = max(0, post_cx - pre_cx)

            cm_source = config.backend_params.get("coupling_map_source", "full")
            cm_edges = len(backend._coupling_map.get_edges()) if backend._coupling_map else 0

            circuit_metrics = CircuitMetrics(
                pre_transpilation_depth=pre_depth,
                pre_transpilation_gate_count=pre_gates,
                pre_transpilation_cx_count=pre_cx,
                post_transpilation_depth=post_depth,
                post_transpilation_gate_count=post_gates,
                post_transpilation_cx_count=post_cx,
                swap_count=swap_count,
                coupling_map_source=cm_source,
                coupling_map_edges=cm_edges // 2,
                transpiler_optimization_level=2,
                num_parameters=ansatz_meta.num_parameters,
            )
            print(f"  Transpiled to coupling map: depth {pre_depth}→{post_depth}, "
                  f"gates {pre_gates}→{post_gates}, SWAPs: ~{swap_count}")

        # Step 4b: Noise config metadata (Phase B)
        noise_config = None
        bp = config.backend_params
        cal_file = bp.get("noise_model_file") or bp.get("coupling_map_file")
        if cal_file:
            from lumi_hpc_qc.backends.noise_model import get_noise_config_metadata
            noise_config = get_noise_config_metadata(
                cal_file, ham_meta.num_qubits,
                bp.get("noise_channels"),
                bp.get("coupling_map_source", "full"),
            )

        # Determine execution mode from config
        shots = config.backend_params.get("shots", 0)
        mode_label = "statevector (exact)" if shots == 0 else f"shot-based ({shots} shots)"
        print(f"  Precision: {config.precision}, Mode: {mode_label}")
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
            "circuit_metrics": circuit_metrics,
            "noise_config": noise_config,
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

        # Phase B: wire circuit metrics and noise config into tracker
        tracker._circuit_metrics = ctx.get("circuit_metrics")
        tracker._noise_config = ctx.get("noise_config")

        # ── Determine execution parameters from config ──
        shots = config.backend_params.get("shots", 0)

        # ── F2 FIX: Energy evaluation through Backend.run_circuits() ──
        eval_count = [0]
        # V19: determine if measurement stats capture is active
        capture_meas = config.capture_measurement_stats and shots > 0

        def eval_energy(params):
            """Evaluate ⟨ψ(θ)|H|ψ(θ)⟩ via the backend interface.

            For statevector backends (shots=0): uses save_expectation_value
            For shot-based backends (shots>0): uses basis-rotated measurement
            For QPU backends: submits to real hardware

            All routing happens inside Backend.run_circuits() — the workflow
            never touches backend internals.
            """
            param_dict = dict(zip(ansatz.parameters, params))

            job_meta = {}
            if capture_meas:
                job_meta["capture_measurement_stats"] = True

            job = CircuitJob(
                circuits=[ansatz],
                parameters=[param_dict],
                observable=hamiltonian,
                shots=shots,
                metadata=job_meta,
            )

            results = backend.run_circuits([job])
            eval_count[0] += 1

            # V19: write measurement stats to sidecar if returned
            if capture_meas and results[0].metadata.get("measurement_stats"):
                for eval_stats in results[0].metadata["measurement_stats"]:
                    current_iter = len(tracker._iterations)
                    tracker.write_measurement_stats(
                        eval_count[0], current_iter, eval_stats
                    )

            if results[0].energies:
                return results[0].energies[0]
            raise RuntimeError("Backend returned no energy for eval_energy")

        # ── Batched energy evaluation for gradient computation ──
        def eval_energy_batch(params_list):
            """Evaluate energy for multiple parameter sets in one backend call.

            Submits all circuits as a single CircuitJob. The backend
            handles batching internally (Aer batches into one sim.run()
            for statevector; iterates for shot-based).
            """
            param_dicts = [
                dict(zip(ansatz.parameters, p)) for p in params_list
            ]

            job = CircuitJob(
                circuits=[ansatz] * len(params_list),
                parameters=param_dicts,
                observable=hamiltonian,
                shots=shots,
            )

            results = backend.run_circuits([job])
            eval_count[0] += len(params_list)

            if results[0].energies:
                return results[0].energies
            raise RuntimeError("Backend returned no energies for eval_energy_batch")

        # ── Select gradient mode ──
        # Batched gradient is available for any backend that supports it
        # (no longer checks for _sim attribute — F2 fix)
        use_batched = (
            grad_strategy is not None
            and grad_strategy.supports_batching
        )

        # Build gradient function
        grad_fn = None
        if grad_strategy is not None:
            if use_batched:
                def grad_fn(params):
                    shifted = grad_strategy.build_shifted_params(params)
                    energies = eval_energy_batch(shifted)
                    return grad_strategy.assemble_gradient(params, energies)
            else:
                def grad_fn(params):
                    return grad_strategy.compute(eval_energy, params, backend)

        # Print optimizer info
        batch_label = " [BATCHED]" if use_batched else ""
        print(f"\n[Step 6] {'Starting' if start_iteration == 0 else 'Resuming'} VQE optimization")
        print(f"  Optimizer: {config.optimizer}")
        if grad_strategy:
            print(f"  Gradient: {grad_name} ({grad_strategy.circuits_per_gradient} circuits/gradient){batch_label}")
        if shots > 0:
            print(f"  Shots per evaluation: {shots}")
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
            record.iteration += start_iteration
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
        if use_batched:
            print(f"  Gradient mode           : BATCHED ({grad_strategy.circuits_per_gradient} circuits/batch)")
        if shots > 0:
            print(f"  Execution mode          : shot-based ({shots} shots/eval)")
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


# Q4 FIX: VQAWorkflow dead class removed.
# Was a placeholder that raised NotImplementedError for both run() and resume().
# If a generic VQA workflow is needed in the future, it should be implemented
# with actual logic, not registered as a discoverable stub.


class CircuitSubmissionWorkflow(Workflow):
    """Direct circuit execution — no optimization loop.

    Submits one or more circuits to a backend and collects results.
    Useful for:
      - Benchmarking circuits on GPU sim vs QPU
      - Running pre-optimized circuits with fixed parameters
      - Collecting measurement statistics for error analysis
      - Testing Q50 connectivity without VQE overhead

    Config fields used:
      - backend: which backend to submit to
      - backend_params.shots: number of measurement shots (default 1024)
      - model + model_params: builds Hamiltonian for expectation computation
      - ansatz + ansatz_params: builds the circuit
      - output_dir: where to save results JSON
    """
    name = "circuit_submission"

    def run(self, config: ExperimentConfig) -> ExperimentRecord:
        from lumi_hpc_qc.backends.registry import BackendRegistry
        from lumi_hpc_qc.data.experiment import ExperimentTracker
        from lumi_hpc_qc.data.provenance import ProvenanceCollector
        from lumi_hpc_qc.data.timing import TimingTracker
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        from lumi_hpc_qc.types import OptimizeResult

        timer = TimingTracker()
        plugins = PluginRegistry()
        plugins.discover()
        backends = BackendRegistry()
        backends.discover()

        provenance = ProvenanceCollector().capture()
        tracker = ExperimentTracker(config)
        tracker.start(provenance=provenance)

        print("=" * 70)
        print(f"  Circuit Submission — {config.model} / {config.ansatz}")
        print(f"  Experiment: {config.experiment_id}")
        print("=" * 70)

        # Build Hamiltonian
        print(f"\n[Step 1] Building Hamiltonian: {config.model}")
        ham_builder = plugins.get_hamiltonian(config.model)
        hamiltonian, ham_meta = ham_builder.build(config)
        config.num_qubits = ham_meta.num_qubits
        timer.mark("hamiltonian_build")

        # Build ansatz
        print(f"\n[Step 2] Building ansatz: {config.ansatz}")
        ansatz_builder = plugins.get_ansatz(config.ansatz)
        ansatz, ansatz_meta = ansatz_builder.build(ham_meta.num_qubits, config)
        print(f"  Parameters: {ansatz_meta.num_parameters}")
        timer.mark("ansatz_build")

        # Initialize backend
        print(f"\n[Step 3] Initializing backend: {config.backend}")
        backend = backends.get(config.backend, config)
        if ansatz_meta.requires_decomposition:
            ansatz = backend.compile_circuit(ansatz)
        timer.mark("backend_init")

        # Initialize parameters
        np.random.seed(config.initializer_params.get("seed", 42))
        params = np.random.uniform(-np.pi / 4, np.pi / 4, ansatz_meta.num_parameters)
        if config.initializer_params.get("fixed_params") is not None:
            params = np.array(config.initializer_params["fixed_params"])
        print(f"\n[Step 4] Parameters: ||θ|| = {float(np.linalg.norm(params)):.4f}")

        # Bind parameters and submit
        param_dict = dict(zip(ansatz.parameters, params))

        shots = config.backend_params.get("shots", 1024)
        print(f"\n[Step 5] Submitting circuit (shots={shots})")

        job = CircuitJob(
            job_id=config.experiment_id,
            circuits=[ansatz],
            parameters=[param_dict],
            observable=hamiltonian,
            shots=shots,
        )

        timer.mark("submission_start")
        results = backend.run_circuits([job])
        timer.mark("submission_complete")

        # Report results
        result = results[0]
        print(f"\n  Execution time: {result.execution_time_s:.2f}s")
        print(f"  Backend: {result.backend_name}")

        if result.energies:
            energy = result.energies[0]
            print(f"  Energy: {energy:+14.8f}")

            exact = ham_builder.exact_ground_energy(hamiltonian)
            if exact is not None:
                err = abs(energy - exact)
                rel = err / abs(exact) * 100 if abs(exact) > 1e-10 else err * 100
                print(f"  Exact:  {exact:+14.8f}")
                print(f"  Error:  {rel:.4f}%")

        if result.counts:
            top_counts = sorted(result.counts.items(), key=lambda x: -x[1])[:10]
            print(f"  Top measurements:")
            for bitstring, count in top_counts:
                print(f"    {bitstring}: {count} ({count/shots*100:.1f}%)")

        timing = timer.finish()
        print(f"\n{timing.to_human_readable()}")

        from lumi_hpc_qc.types import OptimizeResult
        opt_result = OptimizeResult(
            x=params,
            fun=result.energies[0] if result.energies else 0.0,
            nfev=1,
            nit=1,
        )
        record = tracker.finalize(opt_result, timing, exact_energy=None)
        return record

    def resume(self, checkpoint_path: str, config: ExperimentConfig) -> ExperimentRecord:
        print("  CircuitSubmissionWorkflow does not support resume — re-running.")
        return self.run(config)

    # Q3 FIX: declares all plugins actually used by run()
    def get_required_plugins(self) -> dict[str, str]:
        return {
            "hamiltonian": "required",
            "ansatz": "required",
            "error_mitigation": "optional",
        }
