#!/bin/bash

# Builds HNSW indexes for all GIST dataset sizes.
# Creates required subdirectories if they don't exist.
# Skips any size whose index file already exists.

BUILDINDEX="./build/tests/buildindex"
DATA_ROOT="exectable_data/gist1m"

M=32
EF_CONSTRUCTION=200
THREADS=$(nproc)

# Sizes to build — add or remove entries as needed
SIZES=("250k" "500k" "750k" "1000k")

echo "================================================"
echo "Building GIST Indexes"
echo "  M=$M  ef_construction=$EF_CONSTRUCTION  threads=$THREADS"
echo "================================================"

# Verify executable
if [ ! -f "$BUILDINDEX" ]; then
    echo "ERROR: buildindex not found at $BUILDINDEX"
    echo "Run: cd /workspaces/irange/build && cmake .. && make"
    exit 1
fi

for SIZE in "${SIZES[@]}"; do
    DIR="$DATA_ROOT/$SIZE"
    BASE_BIN="$DIR/gist_base_${SIZE}.bin"
    INDEX_FILE="$DIR/gist_${SIZE}.index"

    echo ""
    echo "------------------------------------------------"
    echo "Size: $SIZE"
    echo "------------------------------------------------"

    # Check that the data file exists
    if [ ! -f "$BASE_BIN" ]; then
        echo "SKIP: base data not found at $BASE_BIN"
        continue
    fi

    # Create subdirectories and ensure they are writable
    mkdir -p "$DIR/groundtruth" "$DIR/query_ranges" "$DIR/results"
    chmod a+w "$DIR" "$DIR/groundtruth" "$DIR/query_ranges" "$DIR/results"

    # Skip if index already exists
    if [ -f "$INDEX_FILE" ]; then
        echo "SKIP: index already exists at $INDEX_FILE"
        continue
    fi

    echo "Building index -> $INDEX_FILE"
    $BUILDINDEX \
        --data_path "$BASE_BIN" \
        --index_file "$INDEX_FILE" \
        --M $M \
        --ef_construction $EF_CONSTRUCTION \
        --threads $THREADS

    echo "Done: $INDEX_FILE"
done

echo ""
echo "================================================"
echo "All GIST indexes built successfully!"
echo "================================================"
echo "Indexes:"
for SIZE in "${SIZES[@]}"; do
    INDEX_FILE="$DATA_ROOT/$SIZE/gist_${SIZE}.index"
    if [ -f "$INDEX_FILE" ]; then
        SIZE_HUMAN=$(du -sh "$INDEX_FILE" | cut -f1)
        echo "  [OK] $INDEX_FILE  ($SIZE_HUMAN)"
    else
        echo "  [--] $DATA_ROOT/$SIZE/gist_${SIZE}.index  (not built)"
    fi
done
echo "================================================"
