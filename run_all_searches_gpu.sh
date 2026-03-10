#!/bin/bash

# Script to run GPU searches on all built indexes
# Continues to next dataset even if one fails

echo "================================================"
echo "Running GPU Search Tests on All Indexes"
echo "================================================"

# Default parameters
M=32
GPU_EXECUTABLE="./cude_version/build/hello"

# Check if GPU executable exists
if [ ! -f "$GPU_EXECUTABLE" ]; then
    echo "Error: GPU executable not found at $GPU_EXECUTABLE"
    echo "Please build the GPU version first: cd cude_version && make"
    exit 1
fi

# Track results
declare -a SUCCESSES=()
declare -a FAILURES=()

# Function to run GPU search and track result
run_gpu_search() {
    local dataset_name="$1"
    shift
    local args=("$@")
    
    echo ""
    echo "================================================"
    echo "GPU Search: $dataset_name"
    echo "================================================"
    
    # Run with timeout and error handling
    if timeout 600 "$GPU_EXECUTABLE" "${args[@]}" --M $M; then
        echo "✓ $dataset_name completed successfully"
        SUCCESSES+=("$dataset_name")
    else
        local exit_code=$?
        echo "✗ $dataset_name failed (exit code: $exit_code)"
        FAILURES+=("$dataset_name")
    fi
}

# ============================================
# GIST 250k Index
# ============================================
run_gpu_search "GIST 250k" \
    --data_path exectable_data/gist1m/250k/gist_base_250k.bin \
    --query_path exectable_data/gist1m/250k/gist_query_250k.bin \
    --range_saveprefix exectable_data/gist1m/250k/query_ranges/query_ranges_250k \
    --groundtruth_saveprefix exectable_data/gist1m/250k/ground_truth/ground_truth_250k \
    --index_file exectable_data/gist1m/250k/gist_250k.index \
    --result_saveprefix exectable_data/gist1m/250k/results/results_250k_gpu

# ============================================
# GIST 500k Index
# ============================================
run_gpu_search "GIST 500k" \
    --data_path exectable_data/gist1m/500k/gist_base_500k.bin \
    --query_path exectable_data/gist1m/500k/gist_query_500k.bin \
    --range_saveprefix exectable_data/gist1m/500k/query_ranges/query_ranges_500k \
    --groundtruth_saveprefix exectable_data/gist1m/500k/ground_truth/ground_truth_500k \
    --index_file exectable_data/gist1m/500k/gist_500k.index \
    --result_saveprefix exectable_data/gist1m/500k/results/results_500k_gpu

# ============================================
# GIST 750k Index
# ============================================
run_gpu_search "GIST 750k" \
    --data_path exectable_data/gist1m/750k/gist_base_750k.bin \
    --query_path exectable_data/gist1m/750k/gist_query_750k.bin \
    --range_saveprefix exectable_data/gist1m/750k/query_ranges/query_ranges_750k \
    --groundtruth_saveprefix exectable_data/gist1m/750k/ground_truth/ground_truth_750k \
    --index_file exectable_data/gist1m/750k/gist_750k.index \
    --result_saveprefix exectable_data/gist1m/750k/results/results_750k_gpu

# ============================================
# Video (YouTube RGB) Index
# ============================================
run_gpu_search "Video (YouTube RGB)" \
    --data_path exectable_data/video/youtube_rgb_sorted.bin \
    --query_path exectable_data/video/youtube_rgb_query.bin \
    --range_saveprefix exectable_data/video/query_ranges/query_ranges \
    --groundtruth_saveprefix exectable_data/video/ground_truth/ground_truth \
    --index_file exectable_data/video/youtube_rgb.index \
    --result_saveprefix exectable_data/video/results/results_gpu

# ============================================
# Audi Index
# ============================================
run_gpu_search "Audi" \
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
