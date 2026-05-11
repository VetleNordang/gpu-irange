#!/bin/bash
# Unified search runner for all modes and datasets.
# Usage: run_searches.sh --mode <mode> [--datasets <key>...]
#
# Modes:
#   cpu_serial    OMP_NUM_THREADS=1,    results → cpu_serial/
#   cpu_parallel  OMP_NUM_THREADS=nproc, results → cpu_parallel/
#   gpu_normal    GPU binary,            results → gpu_normal/
#   gpu_pq        GPU PQ binary,         results → gpu_pq/
#
# Dataset keys:
#   gist250k  gist500k  gist750k  gist1000k
#   video1m   video2m   video4m   video8m
#   audi1m    audi2m    audi4m    audi8m
#   (default: all of the above)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$PROJECT_ROOT"

# ── Argument parsing ───────────────────────────────────────────────────────────
MODE=""
DATASETS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)
            MODE="$2"; shift 2 ;;
        --datasets)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                DATASETS+=("$1"); shift
            done
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --mode <cpu_serial|cpu_parallel|gpu_normal|gpu_pq> [--datasets ...]"
            exit 1
            ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "Usage: $0 --mode <cpu_serial|cpu_parallel|gpu_normal|gpu_pq> [--datasets ...]"
    exit 1
fi

if [[ ${#DATASETS[@]} -eq 0 ]]; then
    DATASETS=(
        gist250k gist500k gist750k gist1000k
        video1m  video2m  video4m  video8m
        audi1m   audi2m   audi4m   audi8m
    )
fi

# ── Result directory for this mode ────────────────────────────────────────────
case "$MODE" in
    cpu_serial)   RESULT_DIR="$DIR_CPU_SERIAL"   ;;
    cpu_parallel) RESULT_DIR="$DIR_CPU_PARALLEL" ;;
    gpu_normal)   RESULT_DIR="$DIR_GPU_NORMAL"   ;;
    gpu_pq)       RESULT_DIR="$DIR_GPU_PQ"       ;;
    *)
        echo "Unknown mode: $MODE"
        echo "Valid modes: cpu_serial cpu_parallel gpu_normal gpu_pq"
        exit 1
        ;;
esac

# ── Binary check ───────────────────────────────────────────────────────────────
case "$MODE" in
    cpu_serial|cpu_parallel)
        if [[ ! -f "$CPU_SEARCH_BIN" ]]; then
            echo "ERROR: CPU binary not found: $CPU_SEARCH_BIN"
            echo "Build with: mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
            exit 1
        fi
        CPU_THREADS=1
        [[ "$MODE" == "cpu_parallel" ]] && CPU_THREADS="${SLURM_CPUS_PER_TASK:-$(nproc)}"
        echo "OMP_NUM_THREADS=$CPU_THREADS"
        ;;
    gpu_normal)
        if [[ ! -f "$GPU_SEARCH_BIN" ]]; then
            echo "ERROR: GPU binary not found: $GPU_SEARCH_BIN"
            echo "Build with: cd cude_version && make optimized_test"
            exit 1
        fi
        ;;
    gpu_pq)
        if [[ ! -f "$GPU_PQ_BIN" ]]; then
            echo "ERROR: GPU PQ binary not found: $GPU_PQ_BIN"
            echo "Build with: cd cude_version && make pq_target"
            exit 1
        fi
        ;;
esac

# ── Dataset path resolution ────────────────────────────────────────────────────
# After calling resolve_dataset KEY, these variables are set:
#   D_DATA  D_QUERY  D_INDEX  D_RANGE  D_GT
#   D_RESULTS_BASE  D_CPU_PREFIX  D_GPU_PREFIX
#   D_PQ_M  D_PQ_MODEL  D_PQ_CODES
resolve_dataset() {
    local key="$1"
    case "$key" in
        gist250k|gist500k|gist750k|gist1000k)
            local size="${key#gist}"
            D_DATA="$DATA_ROOT/gist1m/$size/gist_base_$size.bin"
            D_QUERY="$DATA_ROOT/gist1m/$size/gist_query_$size.bin"
            D_INDEX="$DATA_ROOT/gist1m/$size/gist_$size.index"
            D_RANGE="$DATA_ROOT/gist1m/$size/query_ranges/query_ranges_$size"
            D_GT="$DATA_ROOT/gist1m/$size/groundtruth/groundtruth_$size"
            D_RESULTS_BASE="$DATA_ROOT/gist1m/$size/results"
            D_CPU_PREFIX="results_$size"
            D_GPU_PREFIX="results_${size}_gpu"
            D_PQ_M=320
            D_PQ_MODEL="$DATA_ROOT/gist1m/$size/pq/gist_${size}_pq_m320_nb9.faiss"
            D_PQ_CODES="$DATA_ROOT/gist1m/$size/pq/gist_${size}_pq_codes_m320_nb9.bin"
            ;;
        video1m|video2m|video4m|video8m)
            local size="${key#video}"
            D_DATA="$DATA_ROOT/video/$size/youtube_rgb_$size.bin"
            D_QUERY="$DATA_ROOT/video/$size/youtube_rgb_query.bin"
            D_INDEX="$DATA_ROOT/video/$size/youtube_rgb_$size.index"
            D_RANGE="$DATA_ROOT/video/$size/query_ranges/qr"
            D_GT="$DATA_ROOT/video/$size/groundtruth/gt"
            D_RESULTS_BASE="$DATA_ROOT/video/$size/results"
            D_CPU_PREFIX="results"
            D_GPU_PREFIX="results_gpu"
            D_PQ_M=256
            D_PQ_MODEL="$DATA_ROOT/video/$size/pq/video_${size}_pq_m256_nb9.faiss"
            D_PQ_CODES="$DATA_ROOT/video/$size/pq/video_${size}_pq_codes_m256_nb9.bin"
            ;;
        audi1m|audi2m|audi4m|audi8m)
            local size="${key#audi}"
            D_DATA="$DATA_ROOT/audi/$size/yt_aud_$size.bin"
            D_QUERY="$DATA_ROOT/audi/$size/yt_aud_query.bin"
            D_INDEX="$DATA_ROOT/audi/$size/yt_aud_$size.index"
            D_RANGE="$DATA_ROOT/audi/$size/query_ranges/qr"
            D_GT="$DATA_ROOT/audi/$size/groundtruth/gt"
            D_RESULTS_BASE="$DATA_ROOT/audi/$size/results"
            D_CPU_PREFIX="results"
            D_GPU_PREFIX="results_gpu"
            D_PQ_M=32
            D_PQ_MODEL="$DATA_ROOT/audi/$size/pq/audi_${size}_pq_m32_nb9.faiss"
            D_PQ_CODES="$DATA_ROOT/audi/$size/pq/audi_${size}_pq_codes_m32_nb9.bin"
            ;;
        *)
            echo "Unknown dataset key: $key"
            return 1
            ;;
    esac
}

# ── Result tracking ────────────────────────────────────────────────────────────
declare -a SUCCESSES=()
declare -a FAILURES=()
declare -a SKIPS=()

# ── CPU runner (serial and parallel) ──────────────────────────────────────────
run_cpu() {
    local key="$1"
    resolve_dataset "$key" || { FAILURES+=("$key"); return; }

    echo ""
    echo "=== CPU $MODE: $key ==="

    if [[ ! -f "$D_INDEX" ]]; then
        echo "SKIP: index not found — $D_INDEX"
        SKIPS+=("$key")
        return
    fi

    local out_dir="$D_RESULTS_BASE/$RESULT_DIR"
    mkdir -p "$out_dir"
    local prefix="$out_dir/$D_CPU_PREFIX"
    local t_start=$SECONDS

    if OMP_NUM_THREADS="$CPU_THREADS" "$CPU_SEARCH_BIN" \
            --data_path              "$D_DATA"  \
            --query_path             "$D_QUERY" \
            --index_file             "$D_INDEX" \
            --range_saveprefix       "$D_RANGE" \
            --groundtruth_saveprefix "$D_GT"    \
            --result_saveprefix      "$prefix"  \
            --M "$GRAPH_M"; then
        echo "✓ $key done ($(( SECONDS - t_start ))s) → ${prefix}*.csv"
        SUCCESSES+=("$key")
    else
        echo "✗ $key FAILED ($(( SECONDS - t_start ))s)"
        FAILURES+=("$key")
    fi
}

# ── GPU normal runner ──────────────────────────────────────────────────────────
run_gpu_normal() {
    local key="$1"
    resolve_dataset "$key" || { FAILURES+=("$key"); return; }

    echo ""
    echo "=== GPU normal: $key ==="

    if [[ ! -f "$D_INDEX" ]]; then
        echo "SKIP: index not found — $D_INDEX"
        SKIPS+=("$key")
        return
    fi

    local out_dir="$D_RESULTS_BASE/$RESULT_DIR"
    mkdir -p "$out_dir"
    local prefix="$out_dir/$D_GPU_PREFIX"
    local log_file; log_file=$(mktemp)
    local t_start=$SECONDS

    nvidia-smi --query-gpu=name,memory.free,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

    timeout 3600 "$GPU_SEARCH_BIN" \
            --data_path              "$D_DATA"  \
            --query_path             "$D_QUERY" \
            --index_file             "$D_INDEX" \
            --range_saveprefix       "$D_RANGE" \
            --groundtruth_saveprefix "$D_GT"    \
            --result_saveprefix      "$prefix"  \
            --M "$GRAPH_M" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    local failed=0
    [[ $exit_code -ne 0 ]] && failed=1
    grep -Eqi "out of memory|error:" "$log_file" && failed=1
    ls "${prefix}"*.csv >/dev/null 2>&1 || failed=1
    rm -f "$log_file"

    if [[ $failed -eq 0 ]]; then
        echo "✓ $key done ($(( SECONDS - t_start ))s)"
        SUCCESSES+=("$key")
        local plot_script="$PROJECT_ROOT/python/plots/plot_gpu_vs_cpu.py"
        local plot_env=""; [[ -n "${SLURM_JOB_ID:-}" ]] && plot_env="--env idun"
        if [[ -f "$plot_script" ]]; then
            "${CONDA_PYTHON:-python3}" "$plot_script" --dataset "$key" $plot_env || true
        fi
    else
        echo "✗ $key FAILED"
        FAILURES+=("$key")
    fi
}

# ── GPU PQ runner ──────────────────────────────────────────────────────────────
run_gpu_pq() {
    local key="$1"
    resolve_dataset "$key" || { FAILURES+=("$key"); return; }

    echo ""
    echo "=== GPU PQ: $key ==="

    if [[ ! -f "$D_INDEX" ]]; then
        echo "SKIP: index not found — $D_INDEX"
        SKIPS+=("$key")
        return
    fi

    if [[ ! -f "$D_PQ_MODEL" || ! -f "$D_PQ_CODES" ]]; then
        echo "SKIP: PQ files not found — run make_pq.sh first"
        [[ ! -f "$D_PQ_MODEL" ]] && echo "  missing: $D_PQ_MODEL"
        [[ ! -f "$D_PQ_CODES" ]] && echo "  missing: $D_PQ_CODES"
        SKIPS+=("$key")
        return
    fi

    local out_dir="$D_RESULTS_BASE/$RESULT_DIR"
    mkdir -p "$out_dir"
    local prefix="$out_dir/results_pq"
    local log_file; log_file=$(mktemp)
    local t_start=$SECONDS

    nvidia-smi --query-gpu=name,memory.free,memory.used,memory.total --format=csv,noheader 2>/dev/null || true

    timeout 3600 "$GPU_PQ_BIN" \
            --data_path_comp         "$D_DATA"     \
            --query_path             "$D_QUERY"    \
            --index_path             "$D_INDEX"    \
            --range_saveprefix       "$D_RANGE"    \
            --groundtruth_saveprefix "$D_GT"       \
            --result_saveprefix      "$prefix"     \
            --pq_model_out           "$D_PQ_MODEL" \
            --pq_codes_out           "$D_PQ_CODES" \
            --M_compression_spaces   "$D_PQ_M"     \
            --graph_M "$GRAPH_M" 2>&1 | tee "$log_file"
    local exit_code=${PIPESTATUS[0]}

    local failed=0
    [[ $exit_code -ne 0 ]] && failed=1
    grep -Eqi "out of memory|error:" "$log_file" && failed=1
    ls "${prefix}"*.csv >/dev/null 2>&1 || failed=1
    rm -f "$log_file"

    if [[ $failed -eq 0 ]]; then
        echo "✓ $key done ($(( SECONDS - t_start ))s)"
        SUCCESSES+=("$key")
    else
        echo "✗ $key FAILED"
        FAILURES+=("$key")
    fi
}

# ── Main loop ──────────────────────────────────────────────────────────────────
echo "========================================"
echo "mode:     $MODE  (results → $RESULT_DIR/)"
echo "datasets: ${DATASETS[*]}"
echo "started:  $(date)"
echo "========================================"

for key in "${DATASETS[@]}"; do
    case "$MODE" in
        cpu_serial|cpu_parallel) run_cpu "$key" ;;
        gpu_normal)              run_gpu_normal "$key" ;;
        gpu_pq)                  run_gpu_pq "$key" ;;
    esac
done

# ── Summary ────────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "finished: $(date)"
[[ ${#SUCCESSES[@]} -gt 0 ]] && echo "✓ passed  (${#SUCCESSES[@]}): ${SUCCESSES[*]}"
[[ ${#SKIPS[@]}    -gt 0 ]] && echo "- skipped (${#SKIPS[@]}):    ${SKIPS[*]}"
[[ ${#FAILURES[@]} -gt 0 ]] && echo "✗ failed  (${#FAILURES[@]}): ${FAILURES[*]}"
echo "========================================"

[[ ${#FAILURES[@]} -gt 0 ]] && exit 1
exit 0
