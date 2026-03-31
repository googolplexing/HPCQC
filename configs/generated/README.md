# configs/generated/

This directory contains the committed reference set of generated benchmark
configs for TFIM 4q across all 13 noise/topology modes defined in
RED-SPEC-001.

## What is here

13 configs covering all modes for TFIM 4q:

| Mode | Config file | Optimizer | Noise | Topology |
|---|---|---|---|---|
| noiseless | q50bench_tfim_4q_noiseless.yaml | L-BFGS-B + param_shift | None | Full |
| controlled | q50bench_tfim_4q_controlled.yaml | SPSA | None | Full |
| topology_noiseless | q50bench_tfim_4q_topology_noiseless.yaml | SPSA | None | Q50 |
| noise_1q_only | q50bench_tfim_4q_noise_1q_only.yaml | SPSA | 1q depol | Full |
| noise_2q_only | q50bench_tfim_4q_noise_2q_only.yaml | SPSA | 2q depol | Q50 |
| noise_t1_only | q50bench_tfim_4q_noise_t1_only.yaml | SPSA | T1 | Full |
| noise_t2_only | q50bench_tfim_4q_noise_t2_only.yaml | SPSA | T2 | Full |
| noise_readout_only | q50bench_tfim_4q_noise_readout_only.yaml | SPSA | Readout | Full |
| noise_coherence | q50bench_tfim_4q_noise_coherence.yaml | SPSA | T1+T2 | Full |
| noise_gates | q50bench_tfim_4q_noise_gates.yaml | SPSA | 1q+2q depol | Q50 |
| noise_gates_readout | q50bench_tfim_4q_noise_gates_readout.yaml | SPSA | 1q+2q depol+readout | Q50 |
| noise_full | q50bench_tfim_4q_noise_full.yaml | SPSA | All | Q50 |
| qpu | q50bench_tfim_4q_qpu.yaml | SPSA | Real hardware | Q50 |

These 13 configs are the authoritative reference set. They serve as
both documentation of the benchmark matrix and working configs for
immediate SLURM submission.

## What is NOT here

The full 91-config matrix (13 modes × 7 models) is not committed. Generate
it on demand:

```bash
python scripts/generate_configs.py --output configs/generated/
```

**Do not commit the additional 78 generated configs.** They are regenerable
build artifacts. Only the TFIM 4q reference set (these 13) belongs in the
repo.

## Calibration file

All configs reference `examples/q50_calibration_20260330.json` — the full
53-qubit Q50 calibration (14295 bytes, March 30 2026). Do not use
`q50_calibration_20260326.json` (13-qubit stub, superseded).
