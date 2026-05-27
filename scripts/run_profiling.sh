#!/bin/bash
# Run nvprof profiling for gpu_normal or gpu_pq on a given dataset and suffix.
#
# Usage:
#   bash scripts/run_profiling.sh --mode gpu_normal --dataset audi1m
#   bash scripts/run_profiling.sh --mode gpu_pq     --dataset video1m
#   bash scripts/run_profiling.sh --mode gpu_normal --dataset audi1m --suffixes "2 5 8"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"

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

MODE=""
DATASET=""
SUFFIXES=(2 8)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode)    MODE="$2";    shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --suffixes)
            IFS=' ' read -r -a SUFFIXES <<< "$2"; shift 2 ;;
        *)
            echo "Unknown argument: $1"
            echo "Usage: $0 --mode <gpu_normal|gpu_pq> --dataset <key> [--suffixes \"2 5 8\"]"
            exit 1 ;;
    esac
done

if [[ -z "$MODE" || -z "$DATASET" ]]; then
    echo "Usage: $0 --mode <gpu_normal|gpu_pq> --dataset <key> [--suffixes \"2 5 8\"]"
    exit 1
fi

if [[ "$MODE" != "gpu_normal" && "$MODE" != "gpu_pq" ]]; then
    echo "Mode must be gpu_normal or gpu_pq"
    exit 1
fi

cd "$PROJECT_ROOT"

resolve_dataset "$DATASET" || { echo "Unknown dataset: $DATASET"; exit 1; }

PROFILING_DIR="$D_RESULTS_BASE/profiling"
mkdir -p "$PROFILING_DIR"

case "$MODE" in
    gpu_normal) BIN="$GPU_SEARCH_BIN" ;;
    gpu_pq)     BIN="$GPU_PQ_BIN"     ;;
esac

if [[ ! -f "$BIN" ]]; then
    echo "ERROR: binary not found: $BIN"
    echo "Build first: cd cude_version && make optimized_test  (or make pq_target)"
    exit 1
fi

# ── Detect profiler ───────────────────────────────────────────────────────────
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')
CC=${CC:-80}
if [[ "$CC" -le 80 ]]; then
    PROFILER="nvprof"
else
    PROFILER="ncu"
fi

echo "========================================"
echo "mode:     $MODE"
echo "dataset:  $DATASET"
echo "suffixes: ${SUFFIXES[*]}"
echo "binary:   $BIN"
echo "profiler: $PROFILER  (sm_${CC})"
echo "output:   $PROFILING_DIR"
echo "========================================"

run_normal_args=(
    --data_path              "$D_DATA"
    --query_path             "$D_QUERY"
    --index_file             "$D_INDEX"
    --range_saveprefix       "$D_RANGE"
    --groundtruth_saveprefix "$D_GT"
    --M                      "$GRAPH_M"
)

run_pq_args=(
    --data_path_comp         "$D_DATA"
    --query_path             "$D_QUERY"
    --index_path             "$D_INDEX"
    --range_saveprefix       "$D_RANGE"
    --groundtruth_saveprefix "$D_GT"
    --pq_model_out           "$D_PQ_MODEL"
    --pq_codes_out           "$D_PQ_CODES"
    --M_compression_spaces   "$D_PQ_M"
    --graph_M                "$GRAPH_M"
)

for SUFFIX in "${SUFFIXES[@]}"; do
    OUT_CSV="$PROFILING_DIR/kernel_metrics_${MODE}_${DATASET}_suffix${SUFFIX}.csv"
    echo ""
    echo "--- Profiling suffix $SUFFIX -> $OUT_CSV"

    if [[ "$MODE" == "gpu_normal" ]]; then
        BIN_ARGS=("${run_normal_args[@]}"
            --result_saveprefix "$PROFILING_DIR/results_${SUFFIX}"
            --profile_suffix    "$SUFFIX")
    else
        BIN_ARGS=("${run_pq_args[@]}"
            --result_saveprefix "$PROFILING_DIR/results_pq_${SUFFIX}"
            --profile_suffix    "$SUFFIX")
    fi

    if [[ "$PROFILER" == "nvprof" ]]; then
        nvprof --profile-from-start off --metrics all --csv \
            --log-file "$OUT_CSV" \
            "$BIN" "${BIN_ARGS[@]}"
    else
        # ncu: collect key metrics, output to ncu-rep and also export as CSV
        ncu --target-processes all \
            --profile-from-start no \
            --metrics "sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__t_sector_hit_rate.pct,\
lts__t_sector_hit_rate.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_long_sb_per_warp_active.pct,\
smsp__warp_issue_stalled_wait_per_warp_active.pct,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
lts__average_t_sectors_per_request_op_read.ratio,\
dram__bytes_read.sum.per_second" \
            --csv --log-file "$OUT_CSV" \
            "$BIN" "${BIN_ARGS[@]}"
    fi

    if [[ -f "$OUT_CSV" ]]; then
        echo "✓ saved $OUT_CSV"
    else
        echo "✗ profiling failed for suffix $SUFFIX"
    fi
done

echo ""
echo "========================================"
echo "Done. Results in $PROFILING_DIR"
echo "========================================"
