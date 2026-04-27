#!/bin/bash

# Script to run GPU searches on all built indexes
# Continues to next dataset even if one fails

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR" || exit 1

echo "================================================"
echo "Running GPU Search Tests on All Indexes"
echo "================================================"

# Default parameters
M=32
GPU_EXECUTABLE="$SCRIPT_DIR/cude_version/build/optimized_test"
PLOT_SCRIPT="$SCRIPT_DIR/python/plots/plot_gpu_vs_cpu.py"

# Check if GPU executable exists
if [ ! -f "$GPU_EXECUTABLE" ]; then
    echo "Optimized GPU executable not found at $GPU_EXECUTABLE"
    echo "Building optimized GPU target..."
    if ! make -C "$SCRIPT_DIR/cude_version" optimized_test \
            FAISS_INCLUDE="${FAISS_INCLUDE:-}" \
            FAISS_LIB_PATH="${FAISS_LIB_PATH:-}"; then
        echo "Error: failed to build optimized GPU target"
        exit 1
    fi
fi

# Track results
declare -a SUCCESSES=()
declare -a FAILURES=()

# Function to run GPU search and track result
run_gpu_search() {
    local dataset_name="$1"
    local result_prefix="$2"
    local plot_dataset_key="$3"
    shift
    shift

    # Backward compatibility: allow calls without explicit plot dataset key
    if [[ -z "$plot_dataset_key" || "$plot_dataset_key" == --* ]]; then
        case "$dataset_name" in
            "GIST 250k") plot_dataset_key="gist250k" ;;
            "GIST 500k") plot_dataset_key="gist500k" ;;
            "GIST 750k") plot_dataset_key="gist750k" ;;
            "GIST 1000k") plot_dataset_key="gist1000k" ;;
            "Video (YouTube RGB)") plot_dataset_key="video" ;;
            "Audi") plot_dataset_key="audi" ;;
            *) plot_dataset_key="" ;;
        esac
    else
        shift
    fi

    local args=("$@")
    local log_file
    log_file=$(mktemp)
    
    echo ""
    echo "================================================"
    echo "GPU Search: $dataset_name"
    echo "================================================"
    
    # Run with timeout and capture logs for error scanning
    timeout 3600 "$GPU_EXECUTABLE" "${args[@]}" --M $M 2>&1 | tee "$log_file"
    local cmd_exit=${PIPESTATUS[0]}

    local run_failed=0
    if [ $cmd_exit -ne 0 ]; then
        echo "✗ $dataset_name failed (exit code: $cmd_exit)"
        run_failed=1
    fi

    if grep -Eqi "out of memory|failed to initialize visited array|search kernel error|cannot open|error:" "$log_file"; then
        echo "✗ $dataset_name reported runtime errors in output log"
        run_failed=1
    fi

    if ! ls "${result_prefix}"*.csv >/dev/null 2>&1; then
        echo "✗ $dataset_name produced no result CSV files for prefix: ${result_prefix}"
        run_failed=1
    fi

    if [ $run_failed -eq 0 ]; then
        echo "✓ $dataset_name completed successfully"
        SUCCESSES+=("$dataset_name")
    else
        FAILURES+=("$dataset_name")
    fi

    rm -f "$log_file"

    echo "Running plots after: $dataset_name"
    if [ -f "$PLOT_SCRIPT" ]; then
        if python3 "$PLOT_SCRIPT" --dataset "$plot_dataset_key"; then
            echo "✓ Plot generation completed"
        else
            echo "✗ Plot generation failed (continuing to next dataset)"
        fi
    else
        echo "✗ Plot script not found at $PLOT_SCRIPT (continuing to next dataset)"
    fi
}


# ============================================
# Video (YouTube RGB) Index
# ============================================
run_gpu_search "Video 1m (YouTube RGB)" \
    "exectable_data/video/1m/results/results_gpu" \
    "video" \
    --data_path  exectable_data/video/1m/youtube_rgb_1m.bin \
    --query_path exectable_data/video/1m/youtube_rgb_query.bin \
    --range_saveprefix       exectable_data/video/1m/query_ranges/query_ranges \
    --groundtruth_saveprefix exectable_data/video/1m/ground_truth/ground_truth \
    --index_file             exectable_data/video/1m/youtube_rgb_1m.index \
    --result_saveprefix      exectable_data/video/1m/results/results_gpu

# ============================================
# GIST 250k Index
# ============================================
run_gpu_search "GIST 250k" \
    "exectable_data/gist1m/250k/results/results_250k_gpu" \
    "gist250k" \
    --data_path exectable_data/gist1m/250k/gist_base_250k.bin \
    --query_path exectable_data/gist1m/250k/gist_query_250k.bin \
    --range_saveprefix exectable_data/gist1m/250k/query_ranges/query_ranges_250k \
    --groundtruth_saveprefix exectable_data/gist1m/250k/groundtruth/groundtruth_250k \
    --index_file exectable_data/gist1m/250k/gist_250k.index \
    --result_saveprefix exectable_data/gist1m/250k/results/results_250k_gpu

# ============================================
# GIST 500k Index
# ============================================
run_gpu_search "GIST 500k" \
    "exectable_data/gist1m/500k/results/results_500k_gpu" \
    "gist500k" \
    --data_path exectable_data/gist1m/500k/gist_base_500k.bin \
    --query_path exectable_data/gist1m/500k/gist_query_500k.bin \
    --range_saveprefix exectable_data/gist1m/500k/query_ranges/query_ranges_500k \
    --groundtruth_saveprefix exectable_data/gist1m/500k/groundtruth/groundtruth_500k \
    --index_file exectable_data/gist1m/500k/gist_500k.index \
    --result_saveprefix exectable_data/gist1m/500k/results/results_500k_gpu


# ============================================
# GIST 750k Index
# ============================================
run_gpu_search "GIST 750k" \
    "exectable_data/gist1m/750k/results/results_750k_gpu" \
    "gist750k" \
    --data_path exectable_data/gist1m/750k/gist_base_750k.bin \
    --query_path exectable_data/gist1m/750k/gist_query_750k.bin \
    --range_saveprefix exectable_data/gist1m/750k/query_ranges/query_ranges_750k \
    --groundtruth_saveprefix exectable_data/gist1m/750k/groundtruth/groundtruth_750k \
    --index_file exectable_data/gist1m/750k/gist_750k.index \
    --result_saveprefix exectable_data/gist1m/750k/results/results_750k_gpu


# ============================================
# GIST 1000k Index
# ============================================
run_gpu_search "GIST 1000k" \
    "exectable_data/gist1m/1000k/results/results_1000k_gpu" \
    "gist1000k" \
    --data_path exectable_data/gist1m/1000k/gist_base_1000k.bin \
    --query_path exectable_data/gist1m/1000k/gist_query_1000k.bin \
    --range_saveprefix exectable_data/gist1m/1000k/query_ranges/query_ranges_1000k \
    --groundtruth_saveprefix exectable_data/gist1m/1000k/groundtruth/groundtruth_1000k \
    --index_file exectable_data/gist1m/1000k/gist_1000k.index \
    --result_saveprefix exectable_data/gist1m/1000k/results/results_1000k_gpu


# ============================================
# Audi Index
# ============================================
run_gpu_search "Audi" \
    "exectable_data/audi/results/results_gpu" \
    "audi" \
    --data_path exectable_data/audi/yt_aud_sorted_vec_by_attr.bin \
    --query_path exectable_data/audi/yt_aud_ranged_queries.bin \
    --range_saveprefix exectable_data/audi/query_ranges/query_ranges \
    --groundtruth_saveprefix exectable_data/audi/ground_truth/ground_truth \
    --index_file exectable_data/audi/yt_aud_irangegraph_M32.bin \
    --result_saveprefix exectable_data/audi/results/results_gpu

# ============================================
# Summary
# ============================================
echo ""
echo "================================================"
echo "GPU Search Tests Complete!"
echo "================================================"
echo ""

if [ ${#SUCCESSES[@]} -gt 0 ]; then
    echo "✓ Successful datasets (${#SUCCESSES[@]}):"
    for dataset in "${SUCCESSES[@]}"; do
        echo "    - $dataset"
    done
    echo ""
fi

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "✗ Failed datasets (${#FAILURES[@]}):"
    for dataset in "${FAILURES[@]}"; do
        echo "    - $dataset"
    done
    echo ""
fi

echo "GPU Results saved in:"
echo "  - exectable_data/gist1m/250k/results/results_250k_gpu*.csv"
echo "  - exectable_data/gist1m/500k/results/results_500k_gpu*.csv"
echo "  - exectable_data/gist1m/750k/results/results_750k_gpu*.csv"
echo "  - exectable_data/video/results/results_gpu*.csv"
echo "  - exectable_data/audi/results/results_gpu*.csv"
echo "================================================"

# Exit with error if any dataset failed
if [ ${#FAILURES[@]} -gt 0 ]; then
    exit 1
else
    exit 0
fi
