#!/bin/bash
# Copyright (c) 2026 Michael Mucciardi
# SPDX-License-Identifier: SSPL-1.0
# Quick diagnostic summary of lumi-hpc-qc SLURM output files
#
# Usage:
#   ./tests/vqa_summary.sh slurm_logs/*.o*
#   ./tests/vqa_summary.sh slurm_logs/*17007219
#   for f in 123 456; do ./tests/vqa_summary.sh slurm_logs/*${f}; done

for file in "$@"; do
    base=$(basename "$file")
    [ ! -s "$file" ] && continue

    # For error files: only show if actual errors exist
    if [[ "$base" == *.e* ]]; then
        errors=$(grep -c -i "traceback\|exception\|Error:" "$file" 2>/dev/null)
        if [ "$errors" -gt 0 ]; then
            echo "=== $base === [ERRORS]"
            grep -A2 "Traceback\|Error\|Exception" "$file" | head -20
            echo
        fi
        continue
    fi

    # Skip smoke tests for detailed output
    if [[ "$base" == smoke_test* ]]; then
        if grep -q "SMOKE TEST PASSED" "$file" 2>/dev/null; then
            echo "=== $base === SMOKE TEST PASSED"
        else
            echo "=== $base === SMOKE TEST (no result)"
        fi
        echo
        continue
    fi

    echo "=== $base ==="
    grep -E "(^=== |Experiment ID:|VQE Workflow|Building Hamiltonian:|Exact ground state energy:|Building ansatz:|Parameters:|Gradient compatibility:|Preferred initializer:|Initializing backend:|Precision:|Initializing parameters:|Adiabatic init|interaction_u =|jz =|GD steps|Warm-started|Initial energy:|Initial gradient|WARNING.*perturb|Perturbed|Starting VQE|Optimizer:|Gradient:|Max iterations:|CONVERGENCE|Total iterations|Best energy|Exact ground|Best absolute|Best relative|Circuit eval|NOTE:|TOTAL |PASSED|FAILED|RESULT|Total SLURM wall|Node execution)" "$file"
    echo
done

# ── Summary table ──
ofiles=()
for file in "$@"; do
    base=$(basename "$file")
    # Skip error files and smoke tests
    [[ "$base" == *.e* ]] && continue
    [[ "$base" == smoke_test* ]] && continue
    [ ! -s "$file" ] && continue
    ofiles+=("$file")
done

if [ ${#ofiles[@]} -ge 1 ]; then
    echo "========================================================================"
    printf "  %-22s %-28s %12s %12s %9s %7s\n" "Model" "Experiment ID" "Best E" "Exact E" "Error%" "Time"
    echo "  ---------------------- ---------------------------- ------------ ------------ --------- -------"
    for file in "${ofiles[@]}"; do
        base=$(basename "$file")

        # Extract SLURM job ID from filename
        slurm_id=$(echo "$base" | grep -oP '\.o\K[0-9]+')

        # Extract experiment ID
        exp_id=$(grep "Experiment ID:" "$file" | head -1 | grep -oP 'Experiment ID: \K\S+')

        # Build descriptive model name
        workflow=$(grep "VQE Workflow" "$file" | head -1 | grep -oP 'VQE Workflow — \K.*')
        model_raw=$(echo "$workflow" | cut -d' ' -f1)
        nqubits=$(grep "Qubits:" "$file" | head -1 | grep -oP 'Qubits: \K[0-9]+')

        case "$model_raw" in
            byo)
                byo_desc=$(grep -oiP '(TFIM|Ising|Hubbard|custom)\S*' "$file" | head -1 | tr '[:upper:]' '[:lower:]')
                [ -z "$byo_desc" ] && byo_desc="custom"
                model_name="byo_${byo_desc}_${nqubits:-?}q"
                ;;
            fermi_hubbard)
                dims=$(grep -oP '[0-9]+[x×][0-9]+' "$file" | head -1 | tr '×' 'x')
                model_name="fh_${dims:-?}_${nqubits:-?}q"
                ;;
            molecular)
                mol=$(grep -oP 'Molecular VQE: \K\S+' "$file" | head -1 | tr -d ',')
                model_name="mol_${mol:-?}_${nqubits:-?}q"
                ;;
            qaoa_maxcut)
                nodes=$(grep -oP '[0-9]+ nodes' "$file" | head -1 | grep -oP '[0-9]+')
                model_name="qaoa_${nodes:-?}n_${nqubits:-?}q"
                ;;
            heisenberg)
                dims=$(grep -oP '[0-9]+[x×][0-9]+' "$file" | head -1 | tr '×' 'x')
                model_name="heis_${dims:-?}_${nqubits:-?}q"
                ;;
            *)
                model_name="${base%.o*}"
                ;;
        esac

        # Extract results
        best=$(grep "Best energy" "$file" | tail -1 | grep -oP '[-+]?[0-9]+\.[0-9]+' | head -1)
        exact=$(grep "Exact ground state energy:" "$file" | grep -oP '[-+]?[0-9]+\.[0-9]+' | head -1)
        [ -z "$exact" ] && exact=$(grep "Exact ground state" "$file" | grep -v "energy:" | grep -oP '[-+]?[0-9]+\.[0-9]+' | head -1)
        err=$(grep "Best relative" "$file" | grep -oP '[0-9]+\.[0-9]+' | head -1)
        wall=$(grep "Total SLURM wall" "$file" | grep -oP '[0-9]+' | tail -1)

        # Format wall time
        if [ -n "$wall" ]; then
            if [ "$wall" -ge 3600 ]; then
                wstr="$(( wall / 3600 ))h$(( (wall % 3600) / 60 ))m"
            elif [ "$wall" -ge 60 ]; then
                wstr="$(( wall / 60 ))m$(( wall % 60 ))s"
            else
                wstr="${wall}s"
            fi
        else
            wstr="--"
        fi

        # Status flag
        efile="${file/.o/.e}"
        flag=""
        if [ -f "$efile" ] && grep -qi "traceback" "$efile" 2>/dev/null; then
            flag=" ✗"
        elif [ -n "$err" ]; then
            is_good=$(echo "$err < 10" | bc -l 2>/dev/null)
            [ "$is_good" = "1" ] && flag=" ✓"
        fi

        printf "  %-22s %-28s %12s %12s %8s%% %6s%s\n" \
            "$model_name" "${exp_id:---}" "${best:---}" "${exact:---}" "${err:---}" "$wstr" "$flag"
    done
    echo "========================================================================"
fi
