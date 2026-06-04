#!/bin/bash
# Run CPU parallel search on sift1b 10m, 7 times (1 warmup + 6 real runs).
# Results → executable_data/sift1b/10m/results/cpu_parallel/

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BINARY="$PROJECT_ROOT/build/tests/search"
DATA="$PROJECT_ROOT/executable_data/sift1b/10m/sift_10m.bin"
QUERY="$PROJECT_ROOT/executable_data/sift1b/10m/sift_10m_query.bin"
INDEX="$PROJECT_ROOT/executable_data/sift1b/10m/index_M32_ef400.bin"
RANGE_PREFIX="$PROJECT_ROOT/executable_data/sift1b/10m/query_ranges/"
GT_PREFIX="$PROJECT_ROOT/executable_data/sift1b/10m/groundtruth/"
BASE_DIR="$PROJECT_ROOT/executable_data/sift1b/10m/results/cpu_parallel"

NUM_RUNS=7
GRAPH_M=32
CPU_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"

if [[ ! -f "$BINARY" ]]; then
    echo "ERROR: binary not found: $BINARY"
    echo "Build with: mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
    exit 1
fi

echo "========================================"
echo "dataset:  sift1b 10m"
echo "binary:   $BINARY"
echo "threads:  $CPU_THREADS"
echo "runs:     $NUM_RUNS (1 warmup + $(( NUM_RUNS - 1 )) real)"
echo "started:  $(date)"
echo "========================================"

any_failed=0
t_start=$SECONDS

for (( run=0; run<NUM_RUNS; run++ )); do
    if [[ $run -eq 0 ]]; then
        run_dir="$BASE_DIR/warmup"
        run_label="warmup"
    else
        run_dir="$BASE_DIR/run$run"
        run_label="run$run"
    fi

    mkdir -p "$run_dir"
    prefix="$run_dir/results"
    echo "  [$run_label] → $prefix"

    if ! OMP_NUM_THREADS="$CPU_THREADS" "$BINARY" \
            --data_path              "$DATA"         \
            --query_path             "$QUERY"        \
            --index_file             "$INDEX"        \
            --range_saveprefix       "$RANGE_PREFIX" \
            --groundtruth_saveprefix "$GT_PREFIX"    \
            --result_saveprefix      "$prefix"       \
            --M "$GRAPH_M"; then
        echo "  ✗ $run_label FAILED"
        any_failed=1
    fi
done

echo ""
echo "========================================"
echo "finished: $(date)  ($(( SECONDS - t_start ))s)"
if [[ $any_failed -eq 0 ]]; then
    echo "✓ all runs passed"
else
    echo "✗ some runs failed"
    exit 1
fi
echo "========================================"
