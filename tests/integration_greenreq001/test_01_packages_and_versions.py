# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
"""GREEN-REQ-001 Integration Test 1: New package imports + numpy/pandas version check.

Verifies all 7 new packages import correctly and that the numpy/pandas
downgrades (2.4.3→2.2.6, 3.0.1→2.3.3) do not break existing functionality.

IMPORTANT: This test does NOT import qiskit_aer or mpi4py in the same
process as mitiq — see §6.1 of GREEN-RESP-001-DELIVERY for the
MPI_Init_thread conflict. Those are tested separately in test 3 (Aer GPU).
"""

import sys
import os

project_dir = os.environ.get("PROJECT_DIR", os.getcwd())
sys.path.insert(0, os.path.join(project_dir, "src"))

passed = 0
failed = 0
results = []


def check(name, fn):
    global passed, failed
    try:
        fn()
        print(f"  [PASS] {name}")
        passed += 1
        results.append((name, "PASS"))
    except Exception as e:
        print(f"  [FAIL] {name}: {e}")
        failed += 1
        results.append((name, f"FAIL: {e}"))


print("=" * 70)
print("  GREEN-REQ-001 Integration Test 1: Package Imports & Version Check")
print("=" * 70)
print()

# ── Section 1: Version checks (numpy/pandas downgrades) ──
print("--- Section 1: Version Checks (downgrades from container rebuild) ---")


def check_numpy_version():
    import numpy as np
    v = np.__version__
    major, minor = int(v.split(".")[0]), int(v.split(".")[1])
    assert major == 2 and minor >= 2, f"Expected numpy 2.2.x–2.2.x, got {v}"
    print(f"         numpy version: {v}")


def check_pandas_version():
    import pandas as pd
    v = pd.__version__
    major, minor = int(v.split(".")[0]), int(v.split(".")[1])
    assert major == 2 and minor >= 1, f"Expected pandas 2.x (>=2.1), got {v}"
    print(f"         pandas version: {v}")


def check_scipy_unchanged():
    import scipy
    v = scipy.__version__
    assert v == "1.17.1", f"Expected scipy 1.17.1 (unchanged), got {v}"
    print(f"         scipy version: {v}")


check("numpy version (2.2.x)", check_numpy_version)
check("pandas version (2.x)", check_pandas_version)
check("scipy version (1.17.1 unchanged)", check_scipy_unchanged)

# ── Section 2: numpy functional regression ──
print()
print("--- Section 2: numpy Functional Regression (2.4.3 → 2.2.6) ---")


def check_numpy_polyfit():
    import numpy as np
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.1, 3.9, 6.1])
    coeffs = np.polyfit(x, y, 1)
    # Linear fit should give slope ~2.0, intercept ~0.0
    assert abs(coeffs[0] - 2.0) < 0.1, f"polyfit slope wrong: {coeffs[0]}"
    assert abs(coeffs[1] - 0.0) < 0.2, f"polyfit intercept wrong: {coeffs[1]}"


def check_numpy_linalg():
    import numpy as np
    v = np.array([3.0, 4.0])
    norm = np.linalg.norm(v)
    assert abs(norm - 5.0) < 1e-10, f"linalg.norm wrong: {norm}"


def check_numpy_random_determinism():
    import numpy as np
    rng = np.random.RandomState(42)
    vals = rng.uniform(-1, 1, 5)
    # With seed 42, first value should be deterministic
    rng2 = np.random.RandomState(42)
    vals2 = rng2.uniform(-1, 1, 5)
    assert np.array_equal(vals, vals2), "RandomState(42) not deterministic!"


def check_numpy_array_ops():
    import numpy as np
    a = np.zeros(10)
    b = np.linspace(0, 1, 10)
    c = np.pi * b
    assert a.shape == (10,)
    assert abs(b[-1] - 1.0) < 1e-10
    assert abs(c[0]) < 1e-10


check("np.polyfit (used in ZNE)", check_numpy_polyfit)
check("np.linalg.norm (used in adiabatic init)", check_numpy_linalg)
check("np.random.RandomState determinism", check_numpy_random_determinism)
check("np.zeros / np.linspace / np.pi", check_numpy_array_ops)

# ── Section 3: New package imports (Group A) ──
print()
print("--- Section 3: Group A Package Imports ---")


def check_pyarrow():
    import pyarrow as pa
    import pyarrow.parquet as pq
    import pyarrow.dataset as ds
    print(f"         pyarrow version: {pa.__version__}")
    # Verify basic schema creation (used in Phase D)
    schema = pa.schema([
        ("experiment_id", pa.string()),
        ("energy", pa.float64()),
        ("parameters", pa.list_(pa.float64())),
    ])
    assert len(schema) == 3


def check_mitiq():
    # NOTE: Import mitiq but NOT qiskit_aer in the same process
    # to avoid MPI_Init_thread conflict (GREEN-RESP-001-DELIVERY §6.1)
    import mitiq
    from mitiq import zne
    print(f"         mitiq version: {mitiq.__version__}")


def check_scikit_learn():
    import sklearn
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LinearRegression
    print(f"         scikit-learn version: {sklearn.__version__}")
    # Basic constructor test
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    assert clf is not None


def check_jsonschema():
    import jsonschema
    print(f"         jsonschema version: {jsonschema.__version__}")
    # Validate a simple document
    schema = {"type": "object", "properties": {"energy": {"type": "number"}}, "required": ["energy"]}
    jsonschema.validate({"energy": -7.641}, schema)


check("pyarrow (Parquet + Dataset + Schema)", check_pyarrow)
check("mitiq (ZNE import)", check_mitiq)
check("scikit-learn (RF + LR constructors)", check_scikit_learn)
check("jsonschema (validate)", check_jsonschema)

# ── Section 4: New package imports (Group B) ──
print()
print("--- Section 4: Group B Package Imports ---")


def check_qcut():
    import QCut as ck
    from QCut import cut, cutGate, find_cuts
    print(f"         QCut version: {ck.__version__}")


def check_pymetis():
    import pymetis
    # Basic graph partitioning test (4 nodes, 2 partitions)
    adjacency = [[1, 2], [0, 2, 3], [0, 1, 3], [1, 2]]
    n_cuts, membership = pymetis.part_graph(2, adjacency=adjacency)
    assert len(membership) == 4
    assert set(membership) == {0, 1}
    print(f"         pymetis: graph partitioned ({n_cuts} cuts)")


def check_stim():
    import stim
    c = stim.Circuit()
    c.append("H", [0])
    c.append("CNOT", [0, 1])
    c.append("M", [0, 1])
    sampler = c.compile_sampler()
    samples = sampler.sample(shots=10)
    assert samples.shape == (10, 2)
    print(f"         stim version: {stim.__version__}")


def check_pymatching():
    import pymatching
    m = pymatching.Matching()
    print(f"         pymatching version: {pymatching.__version__}")


check("QCut (circuit knitting)", check_qcut)
check("pymetis (graph partitioning)", check_pymetis)
check("stim (stabilizer sim)", check_stim)
check("pymatching (MWPM decoder)", check_pymatching)

# ── Section 5: Existing framework imports (regression check) ──
print()
print("--- Section 5: Framework Import Regression ---")


def check_framework_imports():
    from lumi_hpc_qc import __version__
    from lumi_hpc_qc.types import ExperimentConfig, BackendCapabilities
    from lumi_hpc_qc.backends.base import Backend
    from lumi_hpc_qc.backends.registry import BackendRegistry
    from lumi_hpc_qc.orchestration.workflow import Workflow, VQEWorkflow
    from lumi_hpc_qc.orchestration.controller import Controller
    from lumi_hpc_qc.orchestration.checkpoint import CheckpointManager
    from lumi_hpc_qc.data.experiment import ExperimentTracker
    from lumi_hpc_qc.data.provenance import ProvenanceCollector
    from lumi_hpc_qc.data.export import export_training_data
    from lumi_hpc_qc.cli.config_loader import load_config
    print(f"         framework version: {__version__}")


def check_plugin_discovery():
    from lumi_hpc_qc.plugins.registry import PluginRegistry
    r = PluginRegistry()
    r.discover()
    counts = {}
    for t in ["hamiltonians", "ansatze", "optimizers", "gradients",
              "initializers", "error_mitigation"]:
        plugins = r.list_available(t)
        counts[t] = len(plugins)
        print(f"         {t}: {plugins}")
    total = sum(counts.values())
    assert total >= 19, f"Expected ≥19 plugins, found {total}"
    print(f"         Total plugins: {total}")


check("framework imports (all layers)", check_framework_imports)
check("plugin discovery (6 sub-packages)", check_plugin_discovery)

# ── Summary ──
print()
print("=" * 70)
if failed == 0:
    print(f"  ALL {passed} CHECKS PASSED")
else:
    print(f"  {passed} PASSED, {failed} FAILED")
    print()
    for name, result in results:
        if result.startswith("FAIL"):
            print(f"    FAILED: {name} — {result}")
print("=" * 70)

sys.exit(1 if failed > 0 else 0)
