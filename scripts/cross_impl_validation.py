# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""Cross-implementation validation — RED-SPEC-001 §7.5

Verifies the framework's exact ground state energy computation against two
independent reference implementations:

  1. TFIM 4q — pure numpy tensor product construction (no Qiskit, no HPCQC)
  2. TFIM 4q — framework's own BYO Hamiltonian plugin via SparsePauliOp

Both must agree with the value printed in every TFIM 4q VQE run:
    Exact ground state energy: -4.75877048

Pass criterion: |framework - reference| < 1e-6 (numerical precision floor)

Usage:
    python3 scripts/cross_impl_validation.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def pauli_matrix(char: str):
    """Return the 2x2 Pauli matrix for a single-qubit operator."""
    import numpy as np
    if char == 'I':
        return np.eye(2, dtype=complex)
    elif char == 'X':
        return np.array([[0, 1], [1, 0]], dtype=complex)
    elif char == 'Y':
        return np.array([[0, -1j], [1j, 0]], dtype=complex)
    elif char == 'Z':
        return np.array([[1, 0], [0, -1]], dtype=complex)
    raise ValueError(f"Unknown Pauli: {char}")


def build_hamiltonian_numpy(pauli_terms: list[tuple[str, float]]) -> "np.ndarray":
    """Build Hamiltonian matrix from Pauli string list using pure numpy.

    Uses Qiskit's little-endian convention: rightmost character = qubit 0.
    H = sum_i coeff_i * (P_{n-1} ⊗ ... ⊗ P_1 ⊗ P_0)

    This is completely independent of Qiskit and HPCQC — uses only numpy
    kronecker products of 2x2 Pauli matrices.
    """
    import numpy as np
    from functools import reduce

    n = len(pauli_terms[0][0])
    dim = 2 ** n
    H = np.zeros((dim, dim), dtype=complex)

    for pauli_str, coeff in pauli_terms:
        # Qiskit little-endian: rightmost = qubit 0
        # Build tensor product from left (qubit n-1) to right (qubit 0)
        # kron(A, B) = A ⊗ B, so we go left-to-right through the string
        matrices = [pauli_matrix(c) for c in pauli_str]  # left=high qubit
        term_matrix = reduce(np.kron, matrices)
        H += coeff * term_matrix

    return H


def run_validation() -> bool:
    import numpy as np

    print("=" * 60)
    print("  Cross-Implementation Validation — RED-SPEC-001 §7.5")
    print("  TFIM 4q: J=1.0, g=1.0, open chain")
    print("=" * 60)
    print()

    # TFIM 4q Pauli terms from examples/byo_tfim_4q_q50.json
    pauli_terms = [
        ("ZZII", -1.0), ("IZZI", -1.0), ("IIZZ", -1.0),
        ("XIII", -1.0), ("IXII", -1.0), ("IIXI", -1.0), ("IIIX", -1.0),
    ]

    framework_exact = -4.75877048  # printed by every TFIM 4q VQE run

    all_pass = True

    # ── Reference 1: pure numpy (no Qiskit, no HPCQC) ────────────────
    print("Reference 1: pure numpy tensor products (Qiskit-independent)")
    H_numpy = build_hamiltonian_numpy(pauli_terms)
    eigenvalues = np.linalg.eigvalsh(H_numpy)
    numpy_ground = float(np.real(eigenvalues[0]))
    numpy_diff = abs(numpy_ground - framework_exact)
    numpy_pass = numpy_diff < 1e-6

    print(f"  numpy ground state:     {numpy_ground:.10f}")
    print(f"  framework ground state: {framework_exact:.10f}")
    print(f"  |difference|:           {numpy_diff:.2e}")
    print(f"  {'PASS' if numpy_pass else 'FAIL'} (threshold: 1e-6)")
    if not numpy_pass:
        all_pass = False
    print()

    # ── Reference 2: framework BYO plugin (SparsePauliOp path) ───────
    print("Reference 2: framework BYO Hamiltonian plugin")
    try:
        from lumi_hpc_qc.plugins.hamiltonians.byo import ByoHamiltonian
        from lumi_hpc_qc.types import ExperimentConfig

        config = ExperimentConfig(
            model="byo",
            model_params={"pauli_list": pauli_terms},
            num_qubits=4,
        )

        builder = ByoHamiltonian()
        hamiltonian, meta = builder.build(config)
        plugin_ground = builder.exact_ground_energy(hamiltonian)

        plugin_diff = abs(plugin_ground - framework_exact)
        plugin_pass = plugin_diff < 1e-6

        print(f"  plugin ground state:    {plugin_ground:.10f}")
        print(f"  framework ground state: {framework_exact:.10f}")
        print(f"  |difference|:           {plugin_diff:.2e}")
        print(f"  {'PASS' if plugin_pass else 'FAIL'} (threshold: 1e-6)")
        if not plugin_pass:
            all_pass = False
    except Exception as e:
        print(f"  FAIL: {e}")
        all_pass = False
    print()

    # ── Reference 3: numpy vs plugin (independent consistency check) ──
    print("Reference 3: numpy result vs plugin result (consistency)")
    cross_diff = abs(numpy_ground - plugin_ground)
    cross_pass = cross_diff < 1e-10
    print(f"  numpy:  {numpy_ground:.10f}")
    print(f"  plugin: {plugin_ground:.10f}")
    print(f"  |difference|: {cross_diff:.2e}")
    print(f"  {'PASS' if cross_pass else 'FAIL'} (threshold: 1e-10)")
    if not cross_pass:
        all_pass = False
    print()

    # ── Summary ───────────────────────────────────────────────────────
    print("=" * 60)
    if all_pass:
        print("CROSS-IMPLEMENTATION VALIDATION: ALL PASS")
        print(f"  Framework exact diagonalisation verified against")
        print(f"  independent numpy implementation and BYO plugin.")
        print(f"  Ground state energy -4.75877048 is correct.")
    else:
        print("CROSS-IMPLEMENTATION VALIDATION: FAILED")
    print("=" * 60)

    return all_pass


if __name__ == "__main__":
    passed = run_validation()
    sys.exit(0 if passed else 1)
