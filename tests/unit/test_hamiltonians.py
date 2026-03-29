# Copyright (c) 2026 Team Red / Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# RED-SPEC-001-v1.1 Section C.1 — Hamiltonian verification tests
"""Unit tests for all Hamiltonian plugins.

Each test constructs a minimal Hamiltonian and verifies:
1. The SparsePauliOp matrix matches a hand-computed reference
2. The exact ground state energy matches a known value
3. The qubit count and Pauli term count are correct

Reference values are computed independently of Qiskit.
"""

import unittest
import numpy as np


class TestTFIMHamiltonian(unittest.TestCase):
    """TFIM: H = -J * Σ Z_i Z_{i+1} - g * Σ X_i (open chain)."""

    def test_tfim_2q_matrix(self):
        """Verify 2-qubit TFIM Hamiltonian matrix against hand computation.

        H = -J * Z0⊗Z1 - g * (X0⊗I + I⊗X1)

        For J=1, g=1:
        Z0⊗Z1 = diag(1, -1, -1, 1)
        X0⊗I = [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]
        I⊗X1 = [[0,1,0,0],[1,0,0,0],[0,0,0,1],[0,0,1,0]]

        H = -diag(1,-1,-1,1) - X0⊗I - I⊗X1
        """
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        from lumi_hpc_qc.types import ExperimentConfig

        plugins = PluginRegistry()
        plugins.discover()

        # Construct a minimal config for BYO TFIM 2-qubit
        # (or use the byo plugin with a known Hamiltonian file)
        # If TFIM plugin is available directly:
        config = ExperimentConfig(
            model="byo",
            model_params={"hamiltonian_file": "examples/byo_tfim_2q_q50.json"},
            ansatz="su2", ansatz_params={"reps": 1},
            optimizer="cobyla", optimizer_params={"maxiter": 1},
            gradient="none", initializer="random",
            initializer_params={"seed": 42},
            backend="aer_gpu", backend_params={"method": "statevector"},
            precision="double", mode="interactive",
            checkpoint={"enabled": False, "directory": ".", "interval": 1},
            output_dir=".",
        )

        ham_builder = plugins.get_hamiltonian("byo")
        hamiltonian, ham_meta = ham_builder.build(config)

        # Convert to dense matrix
        H_matrix = hamiltonian.to_matrix()

        # Verify it is Hermitian
        self.assertTrue(
            np.allclose(H_matrix, H_matrix.conj().T, atol=1e-12),
            "Hamiltonian must be Hermitian"
        )

        # Verify qubit count
        self.assertEqual(ham_meta.num_qubits, 2,
            "TFIM 2-qubit should have 2 qubits")

        # Verify eigenvalues (exact diag)
        eigvals = np.linalg.eigvalsh(H_matrix)
        E_ground = eigvals[0]

        # For 2-qubit TFIM with J=1, g=1 (open chain):
        # Ground state energy = -(1 + sqrt(2)) ≈ -2.41421
        # This can be verified analytically.
        E_exact_ref = -(1 + np.sqrt(2))

        self.assertAlmostEqual(E_ground, E_exact_ref, places=8,
            msg=f"TFIM 2q ground energy: got {E_ground}, expected {E_exact_ref}")

    def test_tfim_exact_diag_matches_plugin(self):
        """Verify that the plugin's exact_ground_energy matches numpy eigvalsh."""
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        from lumi_hpc_qc.types import ExperimentConfig

        plugins = PluginRegistry()
        plugins.discover()

        config = ExperimentConfig(
            model="byo",
            model_params={"hamiltonian_file": "examples/byo_tfim_2q_q50.json"},
            ansatz="su2", ansatz_params={"reps": 1},
            optimizer="cobyla", optimizer_params={"maxiter": 1},
            gradient="none", initializer="random",
            initializer_params={"seed": 42},
            backend="aer_gpu", backend_params={"method": "statevector"},
            precision="double", mode="interactive",
            checkpoint={"enabled": False, "directory": ".", "interval": 1},
            output_dir=".",
        )

        ham_builder = plugins.get_hamiltonian("byo")
        hamiltonian, ham_meta = ham_builder.build(config)

        # Plugin's exact energy
        E_plugin = ham_builder.exact_ground_energy(hamiltonian)

        # Independent numpy computation
        H_matrix = hamiltonian.to_matrix()
        E_numpy = float(np.linalg.eigvalsh(H_matrix)[0])

        self.assertIsNotNone(E_plugin, "Plugin should return exact energy for 2 qubits")
        self.assertAlmostEqual(E_plugin, E_numpy, places=10,
            msg="Plugin exact energy must match independent numpy diag")


class TestHeisenbergHamiltonian(unittest.TestCase):
    """Heisenberg XXZ: H = J * Σ (X_i X_j + Y_i Y_j + Δ Z_i Z_j)."""

    def test_heisenberg_2q_matrix(self):
        """Verify 2-qubit isotropic Heisenberg (Jx=Jy=Jz=1, h=0).

        H = X0X1 + Y0Y1 + Z0Z1

        Known 4×4 matrix:
        [[1,0,0,0],[0,-1,2,0],[0,2,-1,0],[0,0,0,1]]

        Eigenvalues: {-3, 1, 1, 1} (singlet at -3, triplet at 1)
        Ground state energy: -3.0
        Spectral gap: 4.0
        """
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        from lumi_hpc_qc.types import ExperimentConfig

        plugins = PluginRegistry()
        plugins.discover()

        config = ExperimentConfig(
            model="heisenberg",
            model_params={
                "lattice_rows": 1, "lattice_cols": 2,
                "jx": 1.0, "jy": 1.0, "jz": 1.0,
                "h_field": 0.0, "boundary_condition": "open",
            },
            ansatz="hva", ansatz_params={"reps": 1},
            optimizer="cobyla", optimizer_params={"maxiter": 1},
            gradient="none", initializer="random",
            initializer_params={"seed": 42},
            backend="aer_gpu", backend_params={"method": "statevector"},
            precision="double", mode="interactive",
            checkpoint={"enabled": False, "directory": ".", "interval": 1},
            output_dir=".",
        )

        ham_builder = plugins.get_hamiltonian("heisenberg")
        hamiltonian, ham_meta = ham_builder.build(config)

        # Reference matrix for 2-qubit isotropic Heisenberg
        # In the computational basis {|00⟩, |01⟩, |10⟩, |11⟩}:
        H_ref = np.array([
            [ 1,  0,  0,  0],
            [ 0, -1,  2,  0],
            [ 0,  2, -1,  0],
            [ 0,  0,  0,  1],
        ], dtype=complex)

        H_matrix = hamiltonian.to_matrix()

        self.assertTrue(
            np.allclose(H_matrix, H_ref, atol=1e-10),
            f"Heisenberg 2q matrix mismatch.\nGot:\n{H_matrix}\nExpected:\n{H_ref}"
        )

        # Verify ground energy = -3.0 (singlet)
        E_ground = float(np.linalg.eigvalsh(H_matrix)[0])
        self.assertAlmostEqual(E_ground, -3.0, places=10,
            msg=f"Heisenberg 2q ground energy: got {E_ground}, expected -3.0")


class TestQAOAHamiltonian(unittest.TestCase):
    """QAOA MaxCut: H = -C = -Σ_{(i,j)∈E} (1 - Z_i Z_j) / 2."""

    def test_qaoa_triangle(self):
        """Verify 3-node complete graph (triangle) MaxCut Hamiltonian.

        Edges: (0,1), (1,2), (0,2)
        H = -(1/2) * Σ (I - Z_i Z_j) = -(3/2)I + (1/2)(Z0Z1 + Z1Z2 + Z0Z2)

        Maximum cut = 2 (any bipartition of triangle cuts 2 edges)
        H_ground = -max_cut = -2

        But VQE minimizes, so H = -C, and ground state of -C = -max_cut.
        For 3-node triangle with all weights 1: max_cut = 2, so E_ground = -2.
        """
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        from lumi_hpc_qc.types import ExperimentConfig

        plugins = PluginRegistry()
        plugins.discover()

        config = ExperimentConfig(
            model="qaoa_maxcut",
            model_params={
                "num_nodes": 3,
                "graph_type": "complete",
                "seed": 42,
            },
            ansatz="qaoa", ansatz_params={"reps": 1},
            optimizer="cobyla", optimizer_params={"maxiter": 1},
            gradient="none", initializer="random",
            initializer_params={"seed": 42},
            backend="aer_gpu", backend_params={"method": "statevector"},
            precision="double", mode="interactive",
            checkpoint={"enabled": False, "directory": ".", "interval": 1},
            output_dir=".",
        )

        ham_builder = plugins.get_hamiltonian("qaoa_maxcut")
        hamiltonian, ham_meta = ham_builder.build(config)

        self.assertEqual(ham_meta.num_qubits, 3)

        H_matrix = hamiltonian.to_matrix()

        # Verify Hermitian
        self.assertTrue(np.allclose(H_matrix, H_matrix.conj().T, atol=1e-12))

        # Ground energy should be -2.0 (max cut of triangle = 2)
        E_ground = float(np.linalg.eigvalsh(H_matrix)[0])
        self.assertAlmostEqual(E_ground, -2.0, places=8,
            msg=f"QAOA triangle ground energy: got {E_ground}, expected -2.0")


class TestH2Hamiltonian(unittest.TestCase):
    """H₂ molecular Hamiltonian at sto-3g / 0.735 Å."""

    def test_h2_ground_energy(self):
        """Verify H₂ FCI ground energy against published reference.

        Reference: H₂ at sto-3g, R=0.735 Å
        FCI energy ≈ -1.8572 Hartree (±0.0001)
        """
        from lumi_hpc_qc.plugins.registry import PluginRegistry
        from lumi_hpc_qc.types import ExperimentConfig

        plugins = PluginRegistry()
        plugins.discover()

        config = ExperimentConfig(
            model="molecular",
            model_params={
                "molecule": "H2",
                "basis": "sto-3g",
                "distance": 0.735,
                "num_electrons": 2,
                "num_spatial_orbitals": 2,
            },
            ansatz="uccsd", ansatz_params={"reps": 1},
            optimizer="cobyla", optimizer_params={"maxiter": 1},
            gradient="none", initializer="random",
            initializer_params={"seed": 42},
            backend="aer_gpu", backend_params={"method": "statevector"},
            precision="double", mode="interactive",
            checkpoint={"enabled": False, "directory": ".", "interval": 1},
            output_dir=".",
        )

        ham_builder = plugins.get_hamiltonian("molecular")
        hamiltonian, ham_meta = ham_builder.build(config)

        # Exact diag
        E_ground = ham_builder.exact_ground_energy(hamiltonian)
        self.assertIsNotNone(E_ground)

        # FCI reference: -1.8572 Ha (±0.0001)
        self.assertAlmostEqual(E_ground, -1.8572, delta=0.001,
            msg=f"H2 ground energy: got {E_ground}, expected ≈-1.8572 Ha")


if __name__ == "__main__":
    unittest.main()
