# CLAUDE.md — Distributed Quantum Simulation & HPC Core

## 1. System Guardrails & Anti-Hallucination
- **Zero Invention**: Never assume or invent APIs, CMake/build flags, Slurm options, env variables, file layouts, or MPI semantics. Verify with project docs or ask.
- **Correctness First**: Correctness and performance are decoupled. Never alter algorithms for speed without verifying physics invariants first. Slower correct code > faster incorrect code.
- **Explicit Shortcuts**: Do not introduce approximations, precision reductions, or truncations without explicit warning. Default to exact `complex128`.

## 2. Distributed HPC & Scalability
- **Memory Tracking**: Optimize strictly for $O(2^N)$ state-vector memory footprints. Eliminate hidden heap allocations or data copies. Prevent memory bloat across nodes.
- **MPI & ROCm Platform**: Target cluster specifics directly (e.g., LUMI, AMD MI250X, Cray MPI). Isolate distributed node orchestration from local compute kernels (HIP).
- **Empirical Proof**: Never claim a speedup, optimization, or scaling benefit without profiling evidence. If unverified via tools (e.g., `rocprof`), state it explicitly as a hypothesis.

## 3. Execution & Verification Loop
1. **Context**: Declare simulation scope and backend (e.g., "Distributed State Vector, MPI+HIP") before coding.
2. **Regression**: Run or write micro-scale correctness tests ($N \le 5$) verifying physics.
3. **Verify**: Compile, execute tests, and profile memory bounds before declaring success.