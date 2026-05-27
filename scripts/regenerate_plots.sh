#!/bin/bash
# Delete all PNGs in executable_data and regenerate all plots for every dataset.
#
# Usage:
#   bash scripts/regenerate_plots.sh
#   bash scripts/regenerate_plots.sh --env idun --hardware "NVIDIA H200 NVL"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$PROJECT_ROOT"

ENV=""
HARDWARE=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --env)      ENV="$2";      shift 2 ;;
        --hardware) HARDWARE="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

PYTHON="${CONDA_PYTHON:-python3}"
PLOT_DIR="$PROJECT_ROOT/python/plots"

# Build args array cleanly so spaces in --hardware value are handled correctly
EXTRA_ARGS=()
[[ -n "$ENV" ]]      && EXTRA_ARGS+=(--env "$ENV")
[[ -n "$HARDWARE" ]] && EXTRA_ARGS+=(--hardware "$HARDWARE")

# ── 1. Delete all PNGs ────────────────────────────────────────────────────────
echo "Deleting all PNGs in executable_data..."
count=$(find -L "$DATA_ROOT" -name "*.png" | wc -l)
find -L "$DATA_ROOT" -name "*.png" -delete
echo "  deleted $count files"

# ── 2. Re-run aggregate ───────────────────────────────────────────────────────
echo ""
echo "Running aggregate_results.py --all ..."
"$PYTHON" python/aggregate_results.py --all

# ── 3. Run each plot script ───────────────────────────────────────────────────
echo ""
echo "Running plot scripts..."

run_plot() {
    local script="$1"
    echo ""
    echo "  $(basename "$script")"
    "$PYTHON" "$script" "${EXTRA_ARGS[@]}"
}

run_plot "$PLOT_DIR/plot_gpu_vs_cpu.py"
run_plot "$PLOT_DIR/plot_recall_thresholds.py"
run_plot "$PLOT_DIR/plot_hops_dco.py"
run_plot "$PLOT_DIR/plot_memory_comparison.py"

echo ""
echo "========================================"
new_count=$(find -L "$DATA_ROOT" -name "*.png" | wc -l)
echo "Done. Generated $new_count PNGs under executable_data/"
echo "========================================"
