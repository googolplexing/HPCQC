<!-- Copyright (c) 2026 Michael Mucciardi -->
<!-- SPDX-License-Identifier: SSPL-1.0 -->

# lumi-hpc-qc

Modular HPC + Quantum Computing framework for LUMI supercomputer and VTT Q50 quantum processor.

## Architecture

Five-layer design with strict downward-only dependencies:

1. **CLI** — config files, SLURM submission, status monitoring
2. **Orchestration** — workflow engine, job scheduler, checkpoint manager, controller
3. **Backends** — pluggable execution: Aer GPU, Aer CPU (MPS), IQM QPU, QCut
4. **Plugins** — researcher-extensible: Hamiltonians, ansatze, optimizers, gradients, initializers, error mitigation
5. **Data** — experiment tracking, provenance, timing, result storage

## Quick start

```bash
# Install (development mode)
pip install -e ".[dev]"

# Run tests
pytest tests/unit/

# Run an experiment interactively
lumi-vqa run --config configs/experiment.yaml.example

# Generate and submit a SLURM job
lumi-vqa submit --config configs/experiment.yaml.example

# Check job status
lumi-vqa status --job-id 12345

# Resume from checkpoint
lumi-vqa resume --config configs/experiment.yaml.example --experiment-id abc123_12345
```

## Adding researcher plugins

Each plugin type has an abstract base class. Implement it, place the file in the right directory, and it's automatically discovered.

### New Hamiltonian

```python
# src/lumi_hpc_qc/plugins/hamiltonians/my_model.py
from lumi_hpc_qc.plugins.hamiltonians.base import HamiltonianBuilder
from lumi_hpc_qc.types import ExperimentConfig, HamiltonianMetadata

class MyModelHamiltonian(HamiltonianBuilder):
    name = "my_model"
    description = "My custom physics model"

    def build(self, config):
        # Build and return (SparsePauliOp, HamiltonianMetadata)
        ...

    def exact_ground_energy(self, hamiltonian):
        # Return exact energy or None
        ...

    def adiabatic_parameter_name(self):
        return None  # or "my_coupling" if adiabatic init applies

    def build_at_parameter(self, value, config):
        ...
```

Then use it: `model: my_model` in your YAML config.

### New ansatz, optimizer, gradient, initializer, or error mitigation

Same pattern — inherit from the base class in the relevant sub-package. See existing implementations for examples.

## Execution modes

### Mode A (interactive)

User submits `salloc` or runs directly on a compute node. Workflow runs in-process.

### Mode B (automated)

Single `sbatch` allocates a CPU-only controller node. Controller submits child SLURM jobs to appropriate partitions (standard-g for GPU, standard for CPU/MPS, q_fiqci for QPU). User can disconnect — job continues until completion or walltime.

## Experiment output

Each experiment produces a structured JSON record containing:
- Full config (reproducible)
- All iteration data (energy, parameters, gradients)
- Convergence summary
- Timing breakdown (human-readable + benchmark + ML training formats)
- Provenance (software versions, hardware, git commit, container tag)

## Project status

- Phase 1: Foundation skeleton (current) — types, ABCs, registries, CLI
- Phase 2: Production features — checkpointing, QPU backend, Mode B, error mitigation
- Phase 3: Advanced — QCut circuit cutting, NVIDIA backends, AI data export

## Tested on

- LUMI (CSC, Finland) — AMD MI250X GPUs, `standard-g` partition
- Qiskit 2.3.0, Qiskit Aer 0.17.2
- Singularity container with ROCm + hipBLAS
- VTT Q50 quantum processor (via FiQCI)
