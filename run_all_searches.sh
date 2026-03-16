#!/bin/bash

# Script to run searches on all built indexes (CPU version only)

echo "================================================"
echo "Running Search Tests on All Indexes"
echo "================================================"

# Default parameters
M=32
SEARCH_EXECUTABLE="./build/tests/search"

# Check if search executable exists
if [ ! -f "$SEARCH_EXECUTABLE" ]; then
    echo "Error: Search executable not found at $SEARCH_EXECUTABLE"
    echo "Please run build_all_variations.sh first to build the project"
    exit 1
fi

echo ""
echo "================================================"
echo "Searching GIST 250k Index"
echo "================================================"
if [ ! -f "exectable_data/gist1m/250k/gist_250k.index" ]; then
    echo "SKIP: index not found, run build_gist_indexes.sh first"
else
$SEARCH_EXECUTABLE \
    --data_path exectable_data/gist1m/250k/gist_base_250k.bin \
    --query_path exectable_data/gist1m/250k/gist_query_250k.bin \
    --range_saveprefix exectable_data/gist1m/250k/query_ranges/query_ranges_250k \
    --groundtruth_saveprefix exectable_data/gist1m/250k/groundtruth/groundtruth_250k \
    --index_file exectable_data/gist1m/250k/gist_250k.index \
    --result_saveprefix exectable_data/gist1m/250k/results/results_250k \
    --M $M
fi

echo ""
echo "================================================"
echo "Searching GIST 500k Index"
echo "================================================"
if [ ! -f "exectable_data/gist1m/500k/gist_500k.index" ]; then
    echo "SKIP: index not found, run build_gist_indexes.sh first"
else
$SEARCH_EXECUTABLE \
    --data_path exectable_data/gist1m/500k/gist_base_500k.bin \
    --query_path exectable_data/gist1m/500k/gist_query_500k.bin \
    --range_saveprefix exectable_data/gist1m/500k/query_ranges/query_ranges_500k \
    --groundtruth_saveprefix exectable_data/gist1m/500k/groundtruth/groundtruth_500k \
    --index_file exectable_data/gist1m/500k/gist_500k.index \
    --result_saveprefix exectable_data/gist1m/500k/results/results_500k \
    --M $M
fi

echo ""
echo "================================================"
echo "Searching GIST 750k Index"
echo "================================================"
if [ ! -f "exectable_data/gist1m/750k/gist_750k.index" ]; then
    echo "SKIP: index not found, run build_gist_indexes.sh first"
else
$SEARCH_EXECUTABLE \
    --data_path exectable_data/gist1m/750k/gist_base_750k.bin \
    --query_path exectable_data/gist1m/750k/gist_query_750k.bin \
    --range_saveprefix exectable_data/gist1m/750k/query_ranges/query_ranges_750k \
    --groundtruth_saveprefix exectable_data/gist1m/750k/groundtruth/groundtruth_750k \
    --index_file exectable_data/gist1m/750k/gist_750k.index \
    --result_saveprefix exectable_data/gist1m/750k/results/results_750k \
    --M $M
fi

echo ""
echo "================================================"
echo "Searching GIST 1000k Index"
echo "================================================"
if [ ! -f "exectable_data/gist1m/1000k/gist_1000k.index" ]; then
    echo "SKIP: index not found, run build_gist_indexes.sh first"
else
$SEARCH_EXECUTABLE \
    --data_path exectable_data/gist1m/1000k/gist_base_1000k.bin \
    --query_path exectable_data/gist1m/1000k/gist_query_1000k.bin \
    --range_saveprefix exectable_data/gist1m/1000k/query_ranges/query_ranges_1000k \
    --groundtruth_saveprefix exectable_data/gist1m/1000k/groundtruth/groundtruth_1000k \
    --index_file exectable_data/gist1m/1000k/gist_1000k.index \
    --result_saveprefix exectable_data/gist1m/1000k/results/results_1000k \
    --M $M
fi

echo ""
echo "================================================"
echo "Searching Video (YouTube RGB) Index"
echo "================================================"
$SEARCH_EXECUTABLE \
    --data_path exectable_data/video/youtube_rgb_sorted.bin \
    --query_path exectable_data/video/youtube_rgb_query.bin \
    --range_saveprefix exectable_data/video/query_ranges/query_ranges \
    --groundtruth_saveprefix exectable_data/video/ground_truth/ground_truth \
    --index_file exectable_data/video/youtube_rgb.index \
    --result_saveprefix exectable_data/video/results/results \
    --M $M

echo ""
echo "================================================"
echo "Searching Audi Index"
echo "================================================"
$SEARCH_EXECUTABLE \
    --data_path exectable_data/audi/yt_aud_sorted_vec_by_attr.bin \
    --query_path exectable_data/audi/yt_aud_ranged_queries.bin \
    --range_saveprefix exectable_data/audi/query_ranges/query_ranges \
    --groundtruth_saveprefix exectable_data/audi/ground_truth/ground_truth \
    --index_file exectable_data/audi/yt_aud_irangegraph_M32.bin \
    --result_saveprefix exectable_data/audi/results/results \
    --M $M

echo ""
echo "================================================"
echo "Search Tests Complete!"
echo "================================================"
echo "Results saved in:"
echo "  - exectable_data/gist1m/250k/results/results_250k*.csv"
echo "  - exectable_data/gist1m/500k/results/results_500k*.csv"
echo "  - exectable_data/gist1m/750k/results/results_750k*.csv"
echo "  - exectable_data/gist1m/1000k/results/results_1000k*.csv"
echo "  - exectable_data/video/results/results*.csv"
echo "  - exectable_data/audi/results/results*.csv"
echo "================================================"
