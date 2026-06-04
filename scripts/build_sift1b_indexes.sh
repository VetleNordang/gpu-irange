#!/bin/bash
set -e

BINARY="~/irange/build/tests/buildindex"
DATA_ROOT="~/irange/executable_data/sift1b"
SIZES=("10m" "30m" "50m" "100m")
M=32
EF=400
THREADS=$(nproc)

for size in "${SIZES[@]}"; do
    data_path="$DATA_ROOT/$size/sift_${size}.bin"
    index_file="$DATA_ROOT/$size/sift_${size}_index.bin"

    echo "=== Building index for $size ==="
    echo "    data:  $data_path"
    echo "    index: $index_file"

    "$BINARY" \
        --data_path "$data_path" \
        --index_file "$index_file" \
        --M $M \
        --ef_construction $EF \
        --threads $THREADS

    echo "    Done: $index_file"
done

echo ""
echo "All indexes built."
