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
if [ ! -f "executable_data/gist1m/250k/gist_250k.index" ]; then
    echo "SKIP: index not found, run build_gist_indexes.sh first"
else
$SEARCH_EXECUTABLE \
    --data_path executable_data/gist1m/250k/gist_base_250k.bin \
    --query_path executable_data/gist1m/250k/gist_query_250k.bin \
    --range_saveprefix executable_data/gist1m/250k/query_ranges/query_ranges_250k \
    --groundtruth_saveprefix executable_data/gist1m/250k/groundtruth/groundtruth_250k \
    --index_file executable_data/gist1m/250k/gist_250k.index \
    --result_saveprefix executable_data/gist1m/250k/results/results_250k \
    --M $M
fi

echo ""
echo "================================================"
echo "Searching GIST 500k Index"
echo "================================================"
if [ ! -f "executable_data/gist1m/500k/gist_500k.index" ]; then
    echo "SKIP: index not found, run build_gist_indexes.sh first"
else
$SEARCH_EXECUTABLE \
    --data_path executable_data/gist1m/500k/gist_base_500k.bin \
    --query_path executable_data/gist1m/500k/gist_query_500k.bin \
    --range_saveprefix executable_data/gist1m/500k/query_ranges/query_ranges_500k \
    --groundtruth_saveprefix executable_data/gist1m/500k/groundtruth/groundtruth_500k \
    --index_file executable_data/gist1m/500k/gist_500k.index \
    --result_saveprefix executable_data/gist1m/500k/results/results_500k \
    --M $M
fi

echo ""
echo "================================================"
echo "Searching GIST 750k Index"
echo "================================================"
if [ ! -f "executable_data/gist1m/750k/gist_750k.index" ]; then
    echo "SKIP: index not found, run build_gist_indexes.sh first"
else
$SEARCH_EXECUTABLE \
    --data_path executable_data/gist1m/750k/gist_base_750k.bin \
    --query_path executable_data/gist1m/750k/gist_query_750k.bin \
    --range_saveprefix executable_data/gist1m/750k/query_ranges/query_ranges_750k \
    --groundtruth_saveprefix executable_data/gist1m/750k/groundtruth/groundtruth_750k \
    --index_file executable_data/gist1m/750k/gist_750k.index \
    --result_saveprefix executable_data/gist1m/750k/results/results_750k \
    --M $M
fi

echo ""
echo "================================================"
echo "Searching GIST 1000k Index"
echo "================================================"
if [ ! -f "executable_data/gist1m/1000k/gist_1000k.index" ]; then
    echo "SKIP: index not found, run build_gist_indexes.sh first"
else
$SEARCH_EXECUTABLE \
    --data_path executable_data/gist1m/1000k/gist_base_1000k.bin \
    --query_path executable_data/gist1m/1000k/gist_query_1000k.bin \
    --range_saveprefix executable_data/gist1m/1000k/query_ranges/query_ranges_1000k \
    --groundtruth_saveprefix executable_data/gist1m/1000k/groundtruth/groundtruth_1000k \
    --index_file executable_data/gist1m/1000k/gist_1000k.index \
    --result_saveprefix executable_data/gist1m/1000k/results/results_1000k \
    --M $M
fi

echo ""
echo "================================================"
echo "Searching Video 1m (YouTube RGB) Index"
echo "================================================"
if [ ! -f "executable_data/video/1m/youtube_rgb_1m.index" ]; then
    echo "SKIP: index not found at executable_data/video/1m/youtube_rgb_1m.index"
else
$SEARCH_EXECUTABLE \
    --data_path executable_data/video/1m/youtube_rgb_1m.bin \
    --query_path executable_data/video/1m/youtube_rgb_query.bin \
    --range_saveprefix executable_data/video/1m/query_ranges/qr \
    --groundtruth_saveprefix executable_data/video/1m/groundtruth/gt \
    --index_file executable_data/video/1m/youtube_rgb_1m.index \
    --result_saveprefix executable_data/video/1m/results/results \
    --M $M
fi

echo ""
echo "================================================"
echo "Searching Audi 1m Index"
echo "================================================"
if [ ! -f "executable_data/audi/1m/yt_aud_1m.index" ]; then
    echo "SKIP: index not found at executable_data/audi/1m/yt_aud_1m.index"
else
$SEARCH_EXECUTABLE \
    --data_path executable_data/audi/1m/yt_aud_1m.bin \
    --query_path executable_data/audi/1m/yt_aud_query.bin \
    --range_saveprefix executable_data/audi/1m/query_ranges/qr \
    --groundtruth_saveprefix executable_data/audi/1m/groundtruth/gt \
    --index_file executable_data/audi/1m/yt_aud_1m.index \
    --result_saveprefix executable_data/audi/1m/results/res \
    --M $M
fi

echo ""
echo "================================================"
echo "Search Tests Complete!"
echo "================================================"
echo "Results saved in:"
echo "  - executable_data/gist1m/250k/results/results_250k*.csv"
echo "  - executable_data/gist1m/500k/results/results_500k*.csv"
echo "  - executable_data/gist1m/750k/results/results_750k*.csv"
echo "  - executable_data/gist1m/1000k/results/results_1000k*.csv"
echo "  - executable_data/video/results/results*.csv"
echo "  - executable_data/audi/results/results*.csv"
echo "================================================"
