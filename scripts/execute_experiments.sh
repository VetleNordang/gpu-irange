#!/bin/bash
# Run one or more experiment modes in sequence.
#
# Usage: ./execute_experiments.sh [modes...] [--datasets <key>...]
#
# Modes:
#   cpu_serial    single-threaded CPU search
#   cpu_parallel  multi-threaded CPU search
#   gpu_normal    GPU search
#   gpu_pq        GPU PQ search
#   all           all four modes (default)
#
# Examples:
#   ./execute_experiments.sh all
#   ./execute_experiments.sh cpu_serial cpu_parallel
#   ./execute_experiments.sh cpu_serial --datasets gist250k gist500k

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="$SCRIPT_DIR/run_searches.sh"

MODES=()
DATASET_ARGS=()
READING_DATASETS=0

for arg in "$@"; do
    if [[ "$arg" == "--datasets" ]]; then
        READING_DATASETS=1
        DATASET_ARGS+=("$arg")
    elif [[ $READING_DATASETS -eq 1 ]]; then
        DATASET_ARGS+=("$arg")
    else
        MODES+=("$arg")
    fi
done

[[ ${#MODES[@]} -eq 0 ]] && MODES=("all")

EXPANDED=()
for m in "${MODES[@]}"; do
    if [[ "$m" == "all" ]]; then
        EXPANDED+=(cpu_serial cpu_parallel gpu_normal gpu_pq)
    else
        EXPANDED+=("$m")
    fi
done

FAILED=0
for mode in "${EXPANDED[@]}"; do
    echo ""
    echo "========================================"
    echo "Starting mode: $mode"
    echo "========================================"
    if ! bash "$RUN" --mode "$mode" "${DATASET_ARGS[@]}"; then
        echo "Mode $mode failed"
        FAILED=1
    fi
done

exit $FAILED
