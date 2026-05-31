#!/usr/bin/env python3
"""Extract the solver's deterministic canonical top_1 placement for the q10
Floquet chain on Q50 (RED ruling #2, Framing A). Read-only: queries the solver,
writes nothing. Run in-container on LUMI (needs rustworkx).

  srun --account=project_462001289 --partition=debug --nodes=1 --ntasks=1 \
       --cpus-per-task=8 --time=00:10:00 \
       ${HPCQC_CPU_WRAPPER} ${HPCQC_CPU_CONTAINER} \
       python3 extract_canonical_placement.py
"""
import json
import sys

# repo root assumed CWD; src on SINGULARITYENV_PYTHONPATH (as in the test runs)
from lumi_hpc_qc.plugins.registry import PluginRegistry
from lumi_hpc_qc.sweep.placement_solver import GeneralPlacementSolver

CAL = "examples/q50_calibration_20260524_08c3c70f.json"

reg = PluginRegistry()
adapter_name = json.load(open(CAL)).get("adapter", "iqm_v2")
adapter = reg.get_calibration_adapter(adapter_name)
device_cal = adapter.load(CAL)

solver = GeneralPlacementSolver()
solver.add_device(device_cal)

# q10 linear Floquet chain: logical edges 0-1, 1-2, ..., 8-9
chain = [(i, i + 1) for i in range(9)]
pls = solver.find_all_placements(
    circuit_edges=chain,
    circuit_qubits=10,
    device_ids=[device_cal.device_id],
    strategy="max_fidelity",
    max_placements=1,
)
if not pls:
    sys.exit("no placements returned for the q10 chain on Q50")

p = pls[0]
order = [p.qubit_mapping[i] for i in range(10)]
print("device_id        :", p.device_id)
print("canonical top_1  :", ",".join(order))   # <- the ordered list to record + pin
print("physical_indices :", p.physical_indices)
print("score            :", p.score)
# A second invocation should print the identical order (determinism check on the spot).
