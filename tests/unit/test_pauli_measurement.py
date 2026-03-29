# Copyright (c) 2026 Team Red / Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# RED-SPEC-001-v1.1 Section C.2 — F1 verification test
# RED-SPEC-001-v1.1 Section C.3 — Decomposition equivalence test
"""Verify that shot-based Pauli measurement matches exact expectation,
and that circuit decomposition preserves the unitary.

F1 verification:
  Constructs a 2-qubit Hamiltonian with X, Y, and Z terms,
  prepares the exact ground state, and compares:
    (a) Exact ⟨H⟩ via save_expectation_value
    (b) Shot-based ⟨H⟩ via basis-rotated measurement (F1 fix)
  The shot-based result must be within 3σ of the exact result.

Decomposition equivalence:
  For each ansatz type, verifies that decompose_for_aer()
  preserves the circuit unitary to within rtol=1e-10.
"""

import unittest
import numpy as np


class TestPauliMeasurementCorrectness(unittest.TestCase):

    def test_mixed_xyz_hamiltonian(self):
        """Test with H = 0.5*ZZ + 0.3*XI + 0.2*YI + 0.1*IX.

        This Hamiltonian has X, Y, AND Z terms — the exact case that
        the original _expectation_from_counts got wrong.
        """
        from qiskit.quantum_info import SparsePauliOp
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator

        # Build Hamiltonian with known X, Y, Z terms
        H = SparsePauliOp.from_list([
            ("ZZ", 0.5),
            ("XI", 0.3),
            ("YI", 0.2),
            ("IX", 0.1),
        ])

        # Find exact ground state via diagonalization
        H_matrix = H.to_matrix()
        eigvals, eigvecs = np.linalg.eigh(H_matrix)
        E_exact = eigvals[0]
        ground_state = eigvecs[:, 0]

        # Build a circuit that prepares the exact ground state
        qc = QuantumCircuit(2)
        qc.initialize(ground_state, [0, 1])

        # (a) Exact expectation via save_expectation_value
        sim = AerSimulator(method="statevector")
        qc_exact = qc.copy()
        qc_exact.save_expectation_value(H, [0, 1], label="energy")
        result_exact = sim.run(qc_exact, shots=0).result()
        E_save_exp = float(np.real(result_exact.data()["energy"]))

        # Verify save_expectation_value matches diagonalization
        self.assertAlmostEqual(E_save_exp, E_exact, places=10,
            msg=f"save_expectation_value ({E_save_exp}) != diag ({E_exact})")

        # (b) Shot-based expectation via F1 fix
        from lumi_hpc_qc.backends.pauli_measurement import (
            build_measurement_circuits,
            expectation_from_grouped_counts,
        )

        N_SHOTS = 100_000
        meas_circuits, meas_groups, identity_e = build_measurement_circuits(
            qc, H, N_SHOTS
        )

        # Run all measurement circuits
        counts_list = []
        for i, mc in enumerate(meas_circuits):
            r = sim.run(mc, shots=N_SHOTS, seed_simulator=42 + i).result()
            counts_list.append(r.get_counts())

        E_shots = expectation_from_grouped_counts(
            counts_list, meas_groups, identity_e, N_SHOTS
        )

        # Statistical tolerance: for N shots, σ ≈ ||H|| / sqrt(N)
        # ||H|| ≤ sum of |coefficients| = 0.5 + 0.3 + 0.2 + 0.1 = 1.1
        # σ ≈ 1.1 / sqrt(100000) ≈ 0.0035
        # 3σ ≈ 0.0104
        sigma_est = 1.1 / np.sqrt(N_SHOTS)
        tolerance = 3 * sigma_est

        self.assertAlmostEqual(E_shots, E_exact, delta=tolerance,
            msg=f"Shot-based E ({E_shots:.6f}) not within 3σ ({tolerance:.6f}) "
                f"of exact E ({E_exact:.6f})")

    def test_pure_z_hamiltonian_unchanged(self):
        """Verify that a pure-Z Hamiltonian still works correctly.

        This is a regression test: the F1 fix must not break Z-only cases.
        """
        from qiskit.quantum_info import SparsePauliOp
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        from lumi_hpc_qc.backends.pauli_measurement import (
            build_measurement_circuits,
            expectation_from_grouped_counts,
        )

        # Pure Z Hamiltonian: H = ZZ + 0.5*ZI + 0.5*IZ
        H = SparsePauliOp.from_list([("ZZ", 1.0), ("ZI", 0.5), ("IZ", 0.5)])

        # Ground state: compute properly
        H_matrix = H.to_matrix()
        E_exact = float(np.linalg.eigvalsh(H_matrix)[0])

        # Prepare ground state
        eigvals, eigvecs = np.linalg.eigh(H_matrix)
        ground_state = eigvecs[:, 0]

        qc = QuantumCircuit(2)
        qc.initialize(ground_state, [0, 1])

        N_SHOTS = 100_000
        meas_circuits, meas_groups, identity_e = build_measurement_circuits(
            qc, H, N_SHOTS
        )

        sim = AerSimulator(method="statevector")
        counts_list = []
        for i, mc in enumerate(meas_circuits):
            r = sim.run(mc, shots=N_SHOTS, seed_simulator=42 + i).result()
            counts_list.append(r.get_counts())

        E_shots = expectation_from_grouped_counts(
            counts_list, meas_groups, identity_e, N_SHOTS
        )

        sigma_est = 2.0 / np.sqrt(N_SHOTS)
        tolerance = 3 * sigma_est

        self.assertAlmostEqual(E_shots, E_exact, delta=tolerance,
            msg=f"Pure-Z shot-based E ({E_shots:.6f}) not within 3σ of exact ({E_exact:.6f})")

    def test_grouping_reduces_circuit_count(self):
        """Verify that QWC grouping produces fewer circuits than Pauli terms."""
        from qiskit.quantum_info import SparsePauliOp
        from lumi_hpc_qc.backends.pauli_measurement import group_commuting_paulis

        # TFIM-like: ZZ, XI, IX — ZZ commutes with neither X term
        # Expected: 2 groups (ZZ alone; XI+IX together since they QWC)
        H = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)])
        groups = group_commuting_paulis(H)
        non_identity = [g for g in groups if not g.get("is_identity")]

        self.assertLessEqual(len(non_identity), 2,
            f"Expected ≤2 measurement groups for ZZ+XI+IX, got {len(non_identity)}")
        self.assertGreaterEqual(len(non_identity), 1,
            "Must have at least 1 measurement group")


class TestDecompositionEquivalence(unittest.TestCase):
    """Verify that decompose_for_aer preserves the circuit unitary."""

    def _check_equivalence(self, ansatz_name, num_qubits, config_params):
        """Helper: build ansatz, decompose, compare unitaries."""
        from qiskit.quantum_info import Operator
        from lumi_hpc_qc.backends.aer_gpu import decompose_for_aer
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        from lumi_hpc_qc.types import ExperimentConfig

        plugins = PluginRegistry()
        plugins.discover()

        config = ExperimentConfig(
            model="byo",
            model_params={"hamiltonian_file": "examples/byo_tfim_2q_q50.json"},
            ansatz=ansatz_name, ansatz_params=config_params,
            optimizer="cobyla", optimizer_params={"maxiter": 1},
            gradient="none", initializer="random",
            initializer_params={"seed": 42},
            backend="aer_gpu", backend_params={"method": "statevector"},
            precision="double", mode="interactive",
            checkpoint={"enabled": False, "directory": ".", "interval": 1},
            output_dir=".",
        )

        ansatz_builder = plugins.get_ansatz(ansatz_name)
        ansatz, meta = ansatz_builder.build(num_qubits, config)

        # Bind parameters to fixed values for unitary comparison
        np.random.seed(42)
        param_values = np.random.uniform(-np.pi, np.pi, meta.num_parameters)
        bound_original = ansatz.assign_parameters(
            dict(zip(ansatz.parameters, param_values))
        )

        # Decompose
        decomposed, rounds = decompose_for_aer(bound_original.copy())

        # Compare unitaries
        U_original = Operator(bound_original)
        U_decomposed = Operator(decomposed)

        self.assertTrue(
            U_original.equiv(U_decomposed, rtol=1e-10),
            f"{ansatz_name} decomposition changed the unitary after {rounds} rounds"
        )

    def test_su2_decomposition(self):
        self._check_equivalence("su2", 2, {"reps": 2, "entanglement": "linear"})

    def test_hva_decomposition(self):
        self._check_equivalence("hva", 2, {"reps": 2, "entanglement": "linear"})

    def test_qaoa_decomposition(self):
        self._check_equivalence("qaoa", 3, {"reps": 1, "entanglement": "linear"})


if __name__ == "__main__":
    unittest.main()
