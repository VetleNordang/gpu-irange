#!/bin/bash

# Builds HNSW indexes for all datasets.
# Usage: bash scripts/build_indexes.sh [all|gist|audi|video]
# Run from project root.

BUILDINDEX="./build/tests/buildindex"
DATA_ROOT="executable_data"

M=32
EF_CONSTRUCTION=200
THREADS=$(nproc)

if [ ! -f "$BUILDINDEX" ]; then
    echo "ERROR: buildindex not found at $BUILDINDEX"
    echo "Run: mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

TARGET="${1:-all}"

echo "================================================"
echo "Building HNSW Indexes  [$TARGET]"
echo "  M=$M  ef_construction=$EF_CONSTRUCTION  threads=$THREADS"
echo "================================================"

build_index() {
    local label="$1"
    local data_file="$2"
    local index_file="$3"

    echo ""
    echo "--- $label ---"

    if [ ! -f "$data_file" ]; then
        echo "SKIP: data not found — $data_file"
        return
    fi

    if [ -f "$index_file" ]; then
        echo "SKIP: already exists — $index_file"
        return
    fi

    echo "Building -> $index_file"
    $BUILDINDEX \
        --data_path "$data_file" \
        --index_file "$index_file" \
        --M $M \
        --ef_construction $EF_CONSTRUCTION \
        --threads $THREADS

    if [ -f "$index_file" ]; then
        echo "Done: $index_file  ($(du -sh "$index_file" | cut -f1))"
    else
        echo "ERROR: index not created — $index_file"
    fi
}

build_gist() {
    for SIZE in 250k 500k 750k 1000k; do
        build_index "gist $SIZE" \
            "$DATA_ROOT/gist1m/$SIZE/gist_base_$SIZE.bin" \
            "$DATA_ROOT/gist1m/$SIZE/gist_${SIZE}.index"
    done
}

build_audi() {
    for SIZE in 1m 2m 4m 8m; do
        build_index "audi $SIZE" \
            "$DATA_ROOT/audi/$SIZE/yt_aud_$SIZE.bin" \
            "$DATA_ROOT/audi/$SIZE/yt_aud_$SIZE.index"
    done
}

build_video() {
    for SIZE in 1m 2m 4m 8m; do
        build_index "video $SIZE" \
            "$DATA_ROOT/video/$SIZE/youtube_rgb_$SIZE.bin" \
            "$DATA_ROOT/video/$SIZE/youtube_rgb_$SIZE.index"
    done
}

case "$TARGET" in
    all)   build_gist; build_audi; build_video ;;
    gist)  build_gist ;;
    audi)  build_audi ;;
    video) build_video ;;
    *)
        echo "ERROR: unknown target '$TARGET'"
        echo "Usage: $0 [all|gist|audi|video]"
        exit 1
        ;;
esac

echo ""
echo "================================================"
echo "Summary"
echo "================================================"
for SIZE in 250k 500k 750k 1000k; do
    f="$DATA_ROOT/gist1m/$SIZE/gist_${SIZE}.index"
    [ -f "$f" ] && echo "  [OK] gist $SIZE" || echo "  [--] gist $SIZE"
done
for SIZE in 1m 2m 4m 8m; do
    f="$DATA_ROOT/audi/$SIZE/yt_aud_$SIZE.index"
    [ -f "$f" ] && echo "  [OK] audi $SIZE" || echo "  [--] audi $SIZE"
done
for SIZE in 1m 2m 4m 8m; do
    f="$DATA_ROOT/video/$SIZE/youtube_rgb_$SIZE.index"
    [ -f "$f" ] && echo "  [OK] video $SIZE" || echo "  [--] video $SIZE"
done
echo "================================================"
