#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# =============================================================================
# lumi-hpc-qc Environment Configuration
# =============================================================================
#
# Central configuration for all container paths, wrappers, and SLURM settings.
# Edit this file ONCE — all sbatch scripts source it automatically.
#
# Users: point these to your own container and wrapper locations.
# =============================================================================

# ── SLURM account ──
export HPCQC_ACCOUNT="${HPCQC_ACCOUNT:-project_462001289}"
# SLURM uses SBATCH_ACCOUNT as the default --account for all sbatch calls
export SBATCH_ACCOUNT="${HPCQC_ACCOUNT}"

# ── Container paths ──
# GPU container (qiskit-aer with ROCm/hipBLAS for MI250X)
export HPCQC_GPU_CONTAINER="${HPCQC_GPU_CONTAINER:-/flash/project_462001289/mucciard/ccpe-extensions-cray-qiskit-aer-patch/output/07-patchelf-fix.sif}"

# CPU container (same image works for CPU, or set a different one)
export HPCQC_CPU_CONTAINER="${HPCQC_CPU_CONTAINER:-${HPCQC_GPU_CONTAINER}}"

# ── Container launch wrappers ──
# These handle Singularity bind mounts and GPU affinity.
# GPU wrapper: sets ROCR_VISIBLE_DEVICES for MI250X GCD mapping
export HPCQC_GPU_WRAPPER="${HPCQC_GPU_WRAPPER:-/flash/project_462001289/mucciard/ccpe-extensions-cray-qiskit-aer-patch/bin/run-singularity-with-gpu-affinity}"

# CPU wrapper: sets up bind mounts without GPU affinity
export HPCQC_CPU_WRAPPER="${HPCQC_CPU_WRAPPER:-/flash/project_462001289/mucciard/ccpe-extensions-cray-qiskit-aer-patch/bin/run-singularity}"

# ── GPU affinity mask ──
# CPU-to-GCD binding mask for LUMI-G nodes (MI250X, 8 GCDs per node)
export HPCQC_GPU_MASK="${HPCQC_GPU_MASK:-mask_cpu:0xfe000000000000,0xfe00000000000000,0xfe0000,0xfe000000,0xfe,0xfe00,0xfe00000000,0xfe0000000000}"

# ── MPICH settings for multi-GPU ──
export MPICH_GPU_IPC_CACHE_MAX_SIZE="${MPICH_GPU_IPC_CACHE_MAX_SIZE:-100}"
export MPICH_GPU_IPC_THRESHOLD="${MPICH_GPU_IPC_THRESHOLD:-524288}"
export MPICH_OFI_NIC_POLICY="${MPICH_OFI_NIC_POLICY:-GPU}"

# ── SLURM partition defaults ──
export HPCQC_GPU_PARTITION="${HPCQC_GPU_PARTITION:-standard-g}"
export HPCQC_CPU_PARTITION="${HPCQC_CPU_PARTITION:-standard}"
export HPCQC_SMALL_PARTITION="${HPCQC_SMALL_PARTITION:-small}"

# ── Derived: project root (directory containing this file) ──
# Scripts source this file, so HPCQC_ROOT is always set correctly
# regardless of where sbatch is invoked from.
export HPCQC_ROOT="${HPCQC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
