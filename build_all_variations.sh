#!/bin/bash

# Build script for CPU version and create indexes for different data sizes

echo "================================================"
echo "Building CPU Version (CMake)"
echo "================================================"

# Create and enter build directory
mkdir -p build
cd build

# Configure and build with CMake (sequential build)
cmake ..
make

cd ..

echo ""
echo "================================================"
echo "Building Indexes for Different Data Sizes"
echo "================================================"

# Default parameters
M=32
EF_CONSTRUCTION=200
THREADS=32

# Build 250k index
echo ""
echo "Building 250k index..."
./build/tests/buildindex \
    --data_path exectable_data/gist1m/gist/gist_base_250k.bin \
    --index_file exectable_data/gist1m/250k/gist_250k.index \
    --M $M \
    --ef_construction $EF_CONSTRUCTION \
    --threads $THREADS

# Build 500k index
echo ""
echo "Building 500k index..."
./build/tests/buildindex \
    --data_path exectable_data/gist1m/gist/gist_base_500k.bin \
    --index_file exectable_data/gist1m/500k/gist_500k.index \
    --M $M \
    --ef_construction $EF_CONSTRUCTION \
    --threads $THREADS

# Build 750k index
echo ""
echo "Building 750k index..."
./build/tests/buildindex \
    --data_path exectable_data/gist1m/gist/gist_base_750k.bin \
    --index_file exectable_data/gist1m/750k/gist_750k.index \
    --M $M \
    --ef_construction $EF_CONSTRUCTION \
    --threads $THREADS

echo ""
echo "================================================"
echo "Build Complete!"
echo "================================================"
echo "Executables built:"
echo "  - build/tests/buildindex"
echo "  - build/tests/search"
echo "  - build/tests/search_multi"
echo ""
echo "Indexes created:"
echo "  - exectable_data/gist1m/250k/gist_250k.index"
echo "  - exectable_data/gist1m/500k/gist_500k.index"
echo "  - exectable_data/gist1m/750k/gist_750k.index"
echo "================================================"
