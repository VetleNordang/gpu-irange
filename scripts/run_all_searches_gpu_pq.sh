#!/bin/bash

# Run GPU PQ searches on all built indexes.
# Continues to next dataset even if one fails.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.." || exit 1

echo "================================================"
echo "Running GPU PQ Search Tests on All Indexes"
echo "================================================"

M=32
PQ_EXECUTABLE="$SCRIPT_DIR/../cude_version/build/hello_pq"

if [ ! -f "$PQ_EXECUTABLE" ]; then
    echo "PQ executable not found at $PQ_EXECUTABLE"
    echo "Building hello_pq target..."
    if ! make -C "$SCRIPT_DIR/../cude_version" pq_target \
            FAISS_INCLUDE="${FAISS_INCLUDE:-}" \
            FAISS_LIB_PATH="${FAISS_LIB_PATH:-}"; then
        echo "Error: failed to build hello_pq"
        exit 1
    fi
fi

declare -a SUCCESSES=()
declare -a FAILURES=()
declare -a SKIPS=()

# run_pq_search <dataset_name> <result_prefix> <pq_model_file> <pq_codes_file> [args...]
# Skips with a message if PQ model or codes files are not present.
run_pq_search() {
    local dataset_name="$1"
    local result_prefix="$2"
    local pq_model="$3"
    local pq_codes="$4"
    shift 4
    local args=("$@")

    echo ""
    echo "================================================"
    echo "GPU PQ Search: $dataset_name"
    echo "================================================"

    if [ ! -f "$pq_model" ] || [ ! -f "$pq_codes" ]; then
        echo "SKIP: PQ files not found — run PQ training first"
        [ ! -f "$pq_model" ] && echo "  missing: $pq_model"
        [ ! -f "$pq_codes" ] && echo "  missing: $pq_codes"
        SKIPS+=("$dataset_name")
        return
    fi

    nvidia-smi --query-gpu=name,memory.free,memory.used,memory.total --format=csv,noheader

    local log_file
    log_file=$(mktemp)
    local t_start=$SECONDS

    timeout 3600 "$PQ_EXECUTABLE" "${args[@]}" 2>&1 | tee "$log_file"
    local cmd_exit=${PIPESTATUS[0]}
    local elapsed=$(( SECONDS - t_start ))

    local run_failed=0
    if [ $cmd_exit -ne 0 ]; then
        echo "✗ $dataset_name failed (exit code: $cmd_exit, elapsed: ${elapsed}s)"
        run_failed=1
    fi

    if grep -Eqi "out of memory|error:" "$log_file"; then
        echo "✗ $dataset_name reported runtime errors (elapsed: ${elapsed}s)"
        run_failed=1
    fi

    if ! ls "${result_prefix}"*.csv >/dev/null 2>&1; then
        echo "✗ $dataset_name produced no result CSV files for prefix: ${result_prefix}"
        run_failed=1
    fi

    rm -f "$log_file"

    if [ $run_failed -eq 0 ]; then
        echo "✓ $dataset_name completed (${elapsed}s)"
        SUCCESSES+=("$dataset_name")
    else
        FAILURES+=("$dataset_name")
    fi
}

# ============================================
# GIST indexes
# ============================================
for size in 250k 500k 750k 1000k; do
    if [ ! -f "executable_data/gist1m/${size}/gist_${size}.index" ]; then
        echo "SKIP: GIST ${size} — index not found"
        continue
    fi
    mkdir -p "executable_data/gist1m/${size}/pq"
    run_pq_search "GIST ${size}" \
        "executable_data/gist1m/${size}/results/results_${size}_pq" \
        "executable_data/gist1m/${size}/pq/gist_${size}_pq_m320_nb9.faiss" \
        "executable_data/gist1m/${size}/pq/gist_${size}_pq_codes_m320_nb9.bin" \
        --data_path_comp executable_data/gist1m/${size}/gist_base_${size}.bin \
        --query_path     executable_data/gist1m/${size}/gist_query_${size}.bin \
        --index_path     executable_data/gist1m/${size}/gist_${size}.index \
        --range_saveprefix       executable_data/gist1m/${size}/query_ranges/query_ranges_${size} \
        --groundtruth_saveprefix executable_data/gist1m/${size}/groundtruth/groundtruth_${size} \
        --result_saveprefix      executable_data/gist1m/${size}/results/results_${size}_pq \
        --pq_model_out   executable_data/gist1m/${size}/pq/gist_${size}_pq_m320_nb9.faiss \
        --pq_codes_out   executable_data/gist1m/${size}/pq/gist_${size}_pq_codes_m320_nb9.bin \
        --M_compression_spaces 320 \
        --graph_M $M
done

# ============================================
# Video (YouTube RGB) indexes — all sizes
# ============================================
for size in 1m 2m 4m 8m; do
    if [ ! -f "executable_data/video/${size}/youtube_rgb_${size}.index" ]; then
        echo "SKIP: Video ${size} — index not found"
        continue
    fi
    mkdir -p "executable_data/video/${size}/pq"
    run_pq_search "Video ${size} (YouTube RGB)" \
        "executable_data/video/${size}/results/results_pq" \
        "executable_data/video/${size}/pq/video_${size}_pq_m256_nb9.faiss" \
        "executable_data/video/${size}/pq/video_${size}_pq_codes_m256_nb9.bin" \
        --data_path_comp executable_data/video/${size}/youtube_rgb_${size}.bin \
        --query_path     executable_data/video/${size}/youtube_rgb_query.bin \
        --index_path     executable_data/video/${size}/youtube_rgb_${size}.index \
        --range_saveprefix       executable_data/video/${size}/query_ranges/qr \
        --groundtruth_saveprefix executable_data/video/${size}/groundtruth/gt \
        --result_saveprefix      executable_data/video/${size}/results/results_pq \
        --pq_model_out   executable_data/video/${size}/pq/video_${size}_pq_m256_nb9.faiss \
        --pq_codes_out   executable_data/video/${size}/pq/video_${size}_pq_codes_m256_nb9.bin \
        --M_compression_spaces 256 \
        --graph_M $M
done

# ============================================
# Audi indexes — all sizes
# ============================================
for size in 1m 2m 4m 8m; do
    if [ ! -f "executable_data/audi/${size}/yt_aud_${size}.index" ]; then
        echo "SKIP: Audi ${size} — index not found"
        continue
    fi
    mkdir -p "executable_data/audi/${size}/pq"
    run_pq_search "Audi ${size}" \
        "executable_data/audi/${size}/results/results_pq" \
        "executable_data/audi/${size}/pq/audi_${size}_pq_m32_nb9.faiss" \
        "executable_data/audi/${size}/pq/audi_${size}_pq_codes_m32_nb9.bin" \
        --data_path_comp executable_data/audi/${size}/yt_aud_${size}.bin \
        --query_path     executable_data/audi/${size}/yt_aud_query.bin \
        --index_path     executable_data/audi/${size}/yt_aud_${size}.index \
        --range_saveprefix       executable_data/audi/${size}/query_ranges/qr \
        --groundtruth_saveprefix executable_data/audi/${size}/groundtruth/gt \
        --result_saveprefix      executable_data/audi/${size}/results/results_pq \
        --pq_model_out   executable_data/audi/${size}/pq/audi_${size}_pq_m32_nb9.faiss \
        --pq_codes_out   executable_data/audi/${size}/pq/audi_${size}_pq_codes_m32_nb9.bin \
        --M_compression_spaces 32 \
        --graph_M $M
done

# ============================================
# Summary
# ============================================
echo ""
echo "================================================"
echo "GPU PQ Search Tests Complete!"
echo "================================================"
echo ""

if [ ${#SUCCESSES[@]} -gt 0 ]; then
    echo "✓ Passed (${#SUCCESSES[@]}):"
    for d in "${SUCCESSES[@]}"; do echo "    $d"; done
    echo ""
fi

if [ ${#SKIPS[@]} -gt 0 ]; then
    echo "- Skipped (${#SKIPS[@]}) — PQ files missing, run training first:"
    for d in "${SKIPS[@]}"; do echo "    $d"; done
    echo ""
fi

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "✗ FAILED (${#FAILURES[@]}):"
    for d in "${FAILURES[@]}"; do echo "    $d"; done
    echo ""
fi

echo "GPU PQ Results saved in:"
echo "  - executable_data/gist1m/{250k,500k,750k,1000k}/results/results_*_pq*.csv"
echo "  - executable_data/video/{1m,2m,4m,8m}/results/results_pq*.csv"
echo "  - executable_data/audi/{1m,2m,4m,8m}/results/results_pq*.csv"
echo "================================================"

if [ ${#FAILURES[@]} -gt 0 ]; then
    exit 1
fi
