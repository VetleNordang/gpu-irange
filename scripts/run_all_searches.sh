#!/bin/bash

# CPU search on all built indexes. Continues past failures and prints a summary.

echo "================================================"
echo "Running CPU Search Tests on All Indexes"
echo "================================================"

M=32
SEARCH_EXECUTABLE="./build/tests/search"

if [ ! -f "$SEARCH_EXECUTABLE" ]; then
    echo "ERROR: search binary not found at $SEARCH_EXECUTABLE"
    exit 1
fi

declare -a SUCCESSES=()
declare -a FAILURES=()
declare -a SKIPS=()

run_cpu_search() {
    local dataset_name="$1"
    local index_file="$2"
    shift 2
    local args=("$@")

    echo ""
    echo "================================================"
    echo "CPU Search: $dataset_name"
    echo "================================================"

    if [ ! -f "$index_file" ]; then
        echo "SKIP: index not found — $index_file"
        SKIPS+=("$dataset_name")
        return
    fi

    local t_start=$SECONDS
    if "$SEARCH_EXECUTABLE" "${args[@]}" --M "$M"; then
        local elapsed=$(( SECONDS - t_start ))
        echo "✓ $dataset_name completed (${elapsed}s)"
        SUCCESSES+=("$dataset_name")
    else
        local exit_code=$?
        local elapsed=$(( SECONDS - t_start ))
        echo "✗ $dataset_name FAILED (exit code: $exit_code, elapsed: ${elapsed}s)"
        FAILURES+=("$dataset_name")
    fi
}

# ── GIST ──────────────────────────────────────────────────────────────────────
for size in 250k 500k 750k 1000k; do
    run_cpu_search "GIST ${size}" \
        "executable_data/gist1m/${size}/gist_${size}.index" \
        --data_path              executable_data/gist1m/${size}/gist_base_${size}.bin \
        --query_path             executable_data/gist1m/${size}/gist_query_${size}.bin \
        --range_saveprefix       executable_data/gist1m/${size}/query_ranges/query_ranges_${size} \
        --groundtruth_saveprefix executable_data/gist1m/${size}/groundtruth/groundtruth_${size} \
        --index_file             executable_data/gist1m/${size}/gist_${size}.index \
        --result_saveprefix      executable_data/gist1m/${size}/results/results_${size}
done

# ── Video ─────────────────────────────────────────────────────────────────────
for size in 1m 2m 4m 8m; do
    run_cpu_search "Video ${size} (YouTube RGB)" \
        "executable_data/video/${size}/youtube_rgb_${size}.index" \
        --data_path              executable_data/video/${size}/youtube_rgb_${size}.bin \
        --query_path             executable_data/video/${size}/youtube_rgb_query.bin \
        --range_saveprefix       executable_data/video/${size}/query_ranges/qr \
        --groundtruth_saveprefix executable_data/video/${size}/groundtruth/gt \
        --index_file             executable_data/video/${size}/youtube_rgb_${size}.index \
        --result_saveprefix      executable_data/video/${size}/results/results
done

# ── Audi ──────────────────────────────────────────────────────────────────────
for size in 1m 2m 4m 8m; do
    run_cpu_search "Audi ${size}" \
        "executable_data/audi/${size}/yt_aud_${size}.index" \
        --data_path              executable_data/audi/${size}/yt_aud_${size}.bin \
        --query_path             executable_data/audi/${size}/yt_aud_query.bin \
        --range_saveprefix       executable_data/audi/${size}/query_ranges/qr \
        --groundtruth_saveprefix executable_data/audi/${size}/groundtruth/gt \
        --index_file             executable_data/audi/${size}/yt_aud_${size}.index \
        --result_saveprefix      executable_data/audi/${size}/results/results
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "================================================"
echo "CPU Search Complete"
echo "================================================"

if [ ${#SUCCESSES[@]} -gt 0 ]; then
    echo "✓ Passed (${#SUCCESSES[@]}):"
    for d in "${SUCCESSES[@]}"; do echo "    $d"; done
fi

if [ ${#SKIPS[@]} -gt 0 ]; then
    echo "- Skipped (${#SKIPS[@]}):"
    for d in "${SKIPS[@]}"; do echo "    $d"; done
fi

if [ ${#FAILURES[@]} -gt 0 ]; then
    echo "✗ FAILED (${#FAILURES[@]}):"
    for d in "${FAILURES[@]}"; do echo "    $d"; done
    echo ""
    echo "Check the log above for each failure — exit code and elapsed time are printed."
    exit 1
fi
