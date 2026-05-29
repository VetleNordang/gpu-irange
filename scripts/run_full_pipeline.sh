#!/bin/bash
# Full pipeline: convert → build indexes → make PQ → search (CPU + GPU)
# Run from project root. Skips steps already done.
# Usage: bash scripts/run_full_pipeline.sh [--runs N] [--datasets key...]
#
# Default: 1 run, all audi+video sizes.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.sh"
cd "$PROJECT_ROOT"

NUM_RUNS=1
DATASETS=(audi1m audi2m audi4m audi6m video1m video2m video4m video6m)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)     NUM_RUNS="$2"; shift 2 ;;
        --datasets) shift; DATASETS=(); while [[ $# -gt 0 && "$1" != --* ]]; do DATASETS+=("$1"); shift; done ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

PYTHON="${CONDA_PYTHON:-python3}"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee "$LOG") 2>&1

echo "========================================"
echo "Full pipeline started: $(date)"
echo "datasets: ${DATASETS[*]}"
echo "runs:     $NUM_RUNS"
echo "log:      $LOG"
echo "========================================"

fail() { echo "ERROR: $1"; exit 1; }
skip() { echo "SKIP: $1"; }
step() { echo ""; echo "════ $1 ════"; }

# ── Step 0: Compile ───────────────────────────────────────────────────────────
step "0. Compile CPU and GPU binaries"

echo "Building CPU binaries ..."
cmake -S "$PROJECT_ROOT" -B "$PROJECT_ROOT/build" -DCMAKE_BUILD_TYPE=Release \
    ${FAISS_INCLUDE_DIR:+-DENABLE_FAISS=ON -DFAISS_INCLUDE_DIR="$FAISS_INCLUDE_DIR" -DFAISS_LIBRARY="$FAISS_LIBRARY" -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath,$FAISS_RPATH -L$FAISS_RPATH"} \
    > /dev/null
cmake --build "$PROJECT_ROOT/build" -j"$(nproc)" \
    || fail "CPU build failed"
echo "  CPU binaries OK"

echo "Building GPU binaries ..."
rm -rf "$PROJECT_ROOT/cude_version/build"
make -C "$PROJECT_ROOT/cude_version" optimized_test pq_target root_target \
    ${CUDA_ARCH_FLAGS:+ARCH="$CUDA_ARCH_FLAGS"} \
    ${FAISS_INCLUDE:+FAISS_INCLUDE="$FAISS_INCLUDE"} \
    ${FAISS_LIB_PATH:+FAISS_LIB_PATH="$FAISS_LIB_PATH"} \
    || fail "GPU build failed"
echo "  GPU binaries OK"

# ── Step 1: Convert TFRecords ─────────────────────────────────────────────────
step "1. Convert TFRecords → .bin"

NEED_AUDIO=0
NEED_VIDEO=0
for key in "${DATASETS[@]}"; do
    case "$key" in
        audi*)  size="${key#audi}"; f="$DATA_ROOT/audi/$size/yt_aud_$size.bin"
                [[ ! -f "$f" ]] && NEED_AUDIO=1 ;;
        video*) size="${key#video}"; f="$DATA_ROOT/video/$size/youtube_rgb_$size.bin"
                [[ ! -f "$f" ]] && NEED_VIDEO=1 ;;
    esac
done

if [[ $NEED_AUDIO -eq 1 ]]; then
    echo "Running convert_yt8m.py --modality audio ..."
    "$PYTHON" "$PROJECT_ROOT/python/prep_attribute/convert_yt8m.py" --modality audio \
        || fail "convert_yt8m.py audio failed"
else
    skip "all audio .bin files already exist"
fi

if [[ $NEED_VIDEO -eq 1 ]]; then
    echo "Running convert_yt8m.py --modality video ..."
    "$PYTHON" "$PROJECT_ROOT/python/prep_attribute/convert_yt8m.py" --modality video \
        || fail "convert_yt8m.py video failed"
else
    skip "all video .bin files already exist"
fi

# ── Step 2: Build indexes ─────────────────────────────────────────────────────
step "2. Build HNSW indexes"

for key in "${DATASETS[@]}"; do
    case "$key" in
        audi*)  size="${key#audi}"
                data="$DATA_ROOT/audi/$size/yt_aud_$size.bin"
                idx="$DATA_ROOT/audi/$size/yt_aud_$size.index" ;;
        video*) size="${key#video}"
                data="$DATA_ROOT/video/$size/youtube_rgb_$size.bin"
                idx="$DATA_ROOT/video/$size/youtube_rgb_$size.index" ;;
    esac

    if [[ -f "$idx" ]]; then
        skip "index exists — $idx"
    elif [[ ! -f "$data" ]]; then
        skip "data missing — $data"
    else
        echo "Building $key index ..."
        "$(dirname "$CPU_SEARCH_BIN")/buildindex" \
            --data_path  "$data" \
            --index_file "$idx"  \
            --M "$GRAPH_M"       \
            --ef_construction 200 \
            --threads "$(nproc)" \
            || fail "buildindex failed for $key"
        echo "  done: $idx"
    fi
done

# ── Step 3: Make PQ ───────────────────────────────────────────────────────────
step "3. Make PQ files"

for key in "${DATASETS[@]}"; do
    case "$key" in
        audi*)  size="${key#audi}"; arg="audi $size" ;;
        video*) size="${key#video}"; arg="video $size" ;;
    esac
    bash "$SCRIPT_DIR/make_pq.sh" $arg || echo "WARN: PQ failed for $key (continuing)"
done

# ── Step 4: CPU search ────────────────────────────────────────────────────────
step "4. CPU search (parallel)"

bash "$SCRIPT_DIR/run_searches.sh" \
    --mode cpu_parallel \
    --runs "$NUM_RUNS"  \
    --datasets "${DATASETS[@]}" \
    || echo "WARN: some CPU searches failed"

# ── Step 5: GPU search ────────────────────────────────────────────────────────
step "5. GPU search (normal)"

if [[ -f "$GPU_SEARCH_BIN" ]]; then
    bash "$SCRIPT_DIR/run_searches.sh" \
        --mode gpu_normal \
        --runs "$NUM_RUNS" \
        --datasets "${DATASETS[@]}" \
        || echo "WARN: some GPU normal searches failed"
else
    skip "GPU binary not found — $GPU_SEARCH_BIN"
fi

step "6. GPU PQ search"

if [[ -f "$GPU_PQ_BIN" ]]; then
    bash "$SCRIPT_DIR/run_searches.sh" \
        --mode gpu_pq \
        --runs "$NUM_RUNS" \
        --datasets "${DATASETS[@]}" \
        || echo "WARN: some GPU PQ searches failed"
else
    skip "GPU PQ binary not found — $GPU_PQ_BIN"
fi

step "7. GPU root search"

if [[ -f "$GPU_ROOT_BIN" ]]; then
    bash "$SCRIPT_DIR/run_searches.sh" \
        --mode gpu_root \
        --runs "$NUM_RUNS" \
        --datasets "${DATASETS[@]}" \
        || echo "WARN: some GPU root searches failed"
else
    skip "GPU root binary not found — $GPU_ROOT_BIN"
fi

echo ""
echo "========================================"
echo "Pipeline complete: $(date)"
echo "Log: $LOG"
echo "========================================"
