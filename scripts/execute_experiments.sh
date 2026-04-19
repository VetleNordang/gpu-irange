#!/bin/bash

###############################################################################
# execute_experiments.sh
#
# Purpose: Execute complete experiment workflow
#
# Usage: ./execute_experiments.sh [mode] [datasets]
#        ./execute_experiments.sh cpu                # Run CPU experiments
#        ./execute_experiments.sh gpu                # Run GPU normal experiments
#        ./execute_experiments.sh pq                 # Run GPU PQ experiments
#        ./execute_experiments.sh all                # Run CPU → GPU → GPU PQ
#
# Datasets (optional, defaults to all):
#        gist250k gist500k gist750k gist1000k audi video
#
# Modes:
#   - cpu:  Original CPU-based search
#   - gpu:  GPU with full embeddings
#   - pq:   GPU with PQ compression
#   - all:  CPU → GPU → GPU PQ (full workflow)
#
# Results saved to: exectable_data/[dataset]/results/[mode]/
# Logs output to: logs/execute_experiments.log
#
###############################################################################

set -e

# Setup paths - resolve to actual location, not symlink
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
CUDE_DIR="$PROJECT_ROOT/cude_version"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/execute_experiments.log"
DATA_ROOT="$PROJECT_ROOT/exectable_data"

# Create logs directory with fallback
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    # Try making it executable-writable
    mkdir -p "$LOG_DIR" 2>/dev/null || true
    # Fallback: write to temp location
    LOG_FILE="/tmp/execute_experiments_$$.log"
fi

# Colors for console output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Functions for logging
log_header() {
    echo -e "${BLUE}╔════════════════════════════════════════╗${NC}" | tee -a "$LOG_FILE"
    echo -e "${BLUE}║${NC} $1" | tee -a "$LOG_FILE"
    echo -e "${BLUE}╚════════════════════════════════════════╝${NC}" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"
}

log_info() {
    echo -e "${YELLOW}ℹ${NC} $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"
}

# Clear previous log
> "$LOG_FILE"

# Start logging
log_header "Execute Experiments Workflow"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Parse arguments
MODE="${1:-all}"
DATASETS_ARG="${@:2}"

# Determine datasets to run
declare -a DATASETS=()

if [ -z "$DATASETS_ARG" ]; then
    # Default to all datasets
    DATASETS=("gist250k" "gist500k" "gist750k" "gist1000k" "audi" "video")
else
    # Use specified datasets
    DATASETS=($DATASETS_ARG)
fi

log_info "Mode: $MODE"
log_info "Datasets: ${DATASETS[@]}"
echo "" | tee -a "$LOG_FILE"

# Verify required files
if [ ! -d "$CUDE_DIR" ]; then
    log_error "CUDA version directory not found: $CUDE_DIR"
    exit 1
fi

log_success "Found CUDA directory"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# PHASE 1: CPU EXPERIMENTS
# ============================================================================
if [ "$MODE" == "cpu" ] || [ "$MODE" == "all" ]; then
    log_header "Phase 1: CPU Experiments"
    
    log_info "Running CPU search on datasets: ${DATASETS[@]}"
    
    cd "$PROJECT_ROOT"
    
    for dataset in "${DATASETS[@]}"; do
        log_info "Processing dataset: $dataset"
        
        # Map dataset names to paths
        case "$dataset" in
            gist250k)
                data_path="$DATA_ROOT/gist1m/250k/gist_base_250k.bin"
                query_path="$DATA_ROOT/gist1m/250k/gist_query_250k.bin"
                index_file="$DATA_ROOT/gist1m/250k/gist_250k.index"
                result_prefix="$DATA_ROOT/gist1m/250k/results/cpu/results_250k"
                dataset_name="gist1m 250k"
                ;;
            gist500k)
                data_path="$DATA_ROOT/gist1m/500k/gist_base_500k.bin"
                query_path="$DATA_ROOT/gist1m/500k/gist_query_500k.bin"
                index_file="$DATA_ROOT/gist1m/500k/gist_500k.index"
                result_prefix="$DATA_ROOT/gist1m/500k/results/cpu/results_500k"
                dataset_name="gist1m 500k"
                ;;
            gist750k)
                data_path="$DATA_ROOT/gist1m/750k/gist_base_750k.bin"
                query_path="$DATA_ROOT/gist1m/750k/gist_query_750k.bin"
                index_file="$DATA_ROOT/gist1m/750k/gist_750k.index"
                result_prefix="$DATA_ROOT/gist1m/750k/results/cpu/results_750k"
                dataset_name="gist1m 750k"
                ;;
            gist1000k)
                data_path="$DATA_ROOT/gist1m/1000k/gist_base_1000k.bin"
                query_path="$DATA_ROOT/gist1m/1000k/gist_query_1000k.bin"
                index_file="$DATA_ROOT/gist1m/1000k/gist_1000k.index"
                result_prefix="$DATA_ROOT/gist1m/1000k/results/cpu/results_1000k"
                dataset_name="gist1m 1000k"
                ;;
            audi)
                log_info "  Skipping audi (CPU implementation may not support this dataset)"
                continue
                ;;
            video)
                data_path="$DATA_ROOT/video/youtube_rgb_sorted.bin"
                query_path="$DATA_ROOT/video/youtube_rgb_query.bin"
                index_file="$DATA_ROOT/video/youtube_rgb.index"
                result_prefix="$DATA_ROOT/video/results/cpu/results"
                dataset_name="Video"
                ;;
            *)
                log_error "Unknown dataset: $dataset"
                continue
                ;;
        esac
        
        # Verify required files exist
        if [ ! -f "$index_file" ]; then
            log_error "  Index file not found: $index_file"
            continue
        fi
        
        # Run search
        log_info "  Running search on $dataset_name..."
        if "$PROJECT_ROOT/build/tests/search" \
            --data_path "$data_path" \
            --query_path "$query_path" \
            --index_file "$index_file" \
            --result_saveprefix "$result_prefix" \
            --M 32 2>&1 | tee -a "$LOG_FILE"; then
            log_success "  CPU experiment completed for $dataset"
        else
            log_error "  CPU experiment failed for $dataset"
        fi
    done
    
    echo "" | tee -a "$LOG_FILE"
fi

# ============================================================================
# PHASE 2: GPU NORMAL EXPERIMENTS
# ============================================================================
if [ "$MODE" == "gpu" ] || [ "$MODE" == "all" ]; then
    log_header "Phase 2: GPU Normal Experiments"
    
    log_info "Building GPU version..."
    cd "$CUDE_DIR"
    
    if make clean 2>&1 | tail -5 | tee -a "$LOG_FILE"; then
        log_success "Cleaned previous build"
    fi
    
    if make 2>&1 | tail -10 | tee -a "$LOG_FILE"; then
        log_success "GPU build completed"
    else
        log_error "GPU build failed"
        exit 1
    fi
    
    echo "" | tee -a "$LOG_FILE"
    
    for dataset in "${DATASETS[@]}"; do
        log_info "Running GPU normal experiments on $dataset..."
        
        # Determine paths based on dataset
        case "$dataset" in
            gist250k|gist500k|gist750k|gist1000k)
                size="${dataset#gist}"
                v_data="$DATA_ROOT/gist1m/$size/gist_base_${size}.bin"
                v_query="$DATA_ROOT/gist1m/$size/gist_query_${size}.bin"
                v_index="$DATA_ROOT/gist1m/$size/gist_${size}.index"
                v_ranges="$DATA_ROOT/gist1m/$size/query_ranges/query_ranges_${size}"
                v_groundtruth="$DATA_ROOT/gist1m/$size/groundtruth/groundtruth_${size}"
                v_results="$DATA_ROOT/gist1m/$size/results/gpu_normal/results_${size}"
                ;;
            audi)
                v_data="$DATA_ROOT/audi/yt_aud_sorted_vec_by_attr.bin"
                v_query="$DATA_ROOT/audi/yt_aud_ranged_queries.bin"
                v_index="$DATA_ROOT/audi/yt_aud_irangegraph_M32.bin"
                v_ranges="$DATA_ROOT/audi/query_ranges/query_ranges"
                v_groundtruth="$DATA_ROOT/audi/ground_truth/ground_truth"
                v_results="$DATA_ROOT/audi/results/gpu_normal/results"
                ;;
            video)
                v_data="$DATA_ROOT/video/youtube_rgb_sorted.bin"
                v_query="$DATA_ROOT/video/youtube_rgb_query.bin"
                v_index="$DATA_ROOT/video/youtube_rgb.index"
                v_ranges="$DATA_ROOT/video/query_ranges/query_ranges"
                v_groundtruth="$DATA_ROOT/video/ground_truth/ground_truth"
                v_results="$DATA_ROOT/video/results/gpu_normal/results"
                ;;
            *)
                log_error "Unknown dataset: $dataset"
                continue
                ;;
        esac
        
        # Verify index exists
        if [ ! -f "$v_index" ]; then
            log_error "  Index file not found: $v_index"
            continue
        fi

        if ./build/hello \
            --data_path "$v_data" \
            --query_path "$v_query" \
            --range_saveprefix "$v_ranges" \
            --groundtruth_saveprefix "$v_groundtruth" \
            --index_file "$v_index" \
            --result_saveprefix "$v_results" \
            --M 32 2>&1 | tee -a "$LOG_FILE"; then
            log_success "  GPU normal experiment completed for $dataset"
        else
            log_error "  GPU normal experiment failed for $dataset"
        fi
    done
    
    echo "" | tee -a "$LOG_FILE"
fi

# ============================================================================
# PHASE 3: GPU PQ EXPERIMENTS
# ============================================================================
if [ "$MODE" == "pq" ] || [ "$MODE" == "all" ]; then
    log_header "Phase 3: GPU PQ Experiments"
    
    log_info "Building GPU PQ version..."
    cd "$CUDE_DIR"
    
    if make pq_target 2>&1 | tail -10 | tee -a "$LOG_FILE"; then
        log_success "GPU PQ build completed"
    else
        log_error "GPU PQ build failed"
        exit 1
    fi
    
    echo "" | tee -a "$LOG_FILE"
        
    for dataset in "${DATASETS[@]}"; do
        log_info "Running GPU PQ experiments on $dataset..."
        
        # Determine paths and parameter M_compression_spaces based on dataset
        case "$dataset" in
            gist250k|gist500k|gist750k|gist1000k)
                size="${dataset#gist}"
                v_data="$DATA_ROOT/gist1m/$size/gist_base_${size}.bin"
                v_query="$DATA_ROOT/gist1m/$size/gist_query_${size}.bin"
                v_index="$DATA_ROOT/gist1m/$size/gist_${size}.index"
                v_ranges="$DATA_ROOT/gist1m/$size/query_ranges/query_ranges_${size}"
                v_groundtruth="$DATA_ROOT/gist1m/$size/groundtruth/groundtruth_${size}"
                v_results="$DATA_ROOT/gist1m/$size/results/gpu_pq/results_pq"
                v_pq_model="$DATA_ROOT/gist1m/$size/pq/gist_${size}_pq_m320_nb9.faiss"
                v_pq_codes="$DATA_ROOT/gist1m/$size/pq/gist_${size}_pq_codes_m320_nb9.bin"
                m_comp=320
                ;;
            audi)
                v_data="$DATA_ROOT/audi/yt_aud_sorted_vec_by_attr.bin"
                v_query="$DATA_ROOT/audi/yt_aud_ranged_queries.bin"
                v_index="$DATA_ROOT/audi/yt_aud_irangegraph_M32.bin"
                v_ranges="$DATA_ROOT/audi/query_ranges/query_ranges"
                v_groundtruth="$DATA_ROOT/audi/ground_truth/ground_truth"
                v_results="$DATA_ROOT/audi/results/gpu_pq/results_pq"
                v_pq_model="$DATA_ROOT/audi/pq/audi_pq_m32_nb9.faiss"
                v_pq_codes="$DATA_ROOT/audi/pq/audi_pq_codes_m32_nb9.bin"
                m_comp=32
                ;;
            video)
                v_data="$DATA_ROOT/video/youtube_rgb_sorted.bin"
                v_query="$DATA_ROOT/video/youtube_rgb_query.bin"
                v_index="$DATA_ROOT/video/youtube_rgb.index"
                v_ranges="$DATA_ROOT/video/query_ranges/query_ranges"
                v_groundtruth="$DATA_ROOT/video/ground_truth/ground_truth"
                v_results="$DATA_ROOT/video/results/gpu_pq/results_pq"
                v_pq_model="$DATA_ROOT/video/pq/video_pq_m256_nb9.faiss"
                v_pq_codes="$DATA_ROOT/video/pq/video_pq_codes_m256_nb9.bin"
                m_comp=256
                ;;
            *)
                log_error "Unknown dataset: $dataset"
                continue
                ;;
        esac
        
        # Verify index exists
        if [ ! -f "$v_index" ]; then
            log_error "  Index file not found: $v_index"
            continue
        fi

        if ./build/hello_pq \
            --data_path_comp "$v_data" \
            --query_path "$v_query" \
            --index_path "$v_index" \
            --result_saveprefix "$v_results" \
            --range_saveprefix "$v_ranges" \
            --groundtruth_saveprefix "$v_groundtruth" \
            --pq_model_out "$v_pq_model" \
            --pq_codes_out "$v_pq_codes" \
            --M_compression_spaces "$m_comp" \
            --graph_M 32 2>&1 | tee -a "$LOG_FILE"; then
            log_success "  GPU PQ experiment completed for $dataset"
        else
            log_error "  GPU PQ experiment failed for $dataset"
        fi
    done
    
    echo "" | tee -a "$LOG_FILE"
fi

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
log_header "Experiments Summary"

log_success "Experiments workflow completed"
echo "" | tee -a "$LOG_FILE"

log_info "Results locations:"
echo "" | tee -a "$LOG_FILE"

if [ "$MODE" == "cpu" ] || [ "$MODE" == "all" ]; then
    echo "CPU Results:" | tee -a "$LOG_FILE"
    for dataset in "${DATASETS[@]}"; do
        case "$dataset" in
            gist250k|gist500k|gist750k|gist1000k)
                size="${dataset#gist}"
                result_dir="$DATA_ROOT/gist1m/$size/results/cpu"
                ;;
            video)
                result_dir="$DATA_ROOT/video/results/cpu"
                ;;
            *)
                continue
                ;;
        esac
        
        if [ -d "$result_dir" ]; then
            count=$(find "$result_dir" -name "*.csv" -type f 2>/dev/null | wc -l)
            echo "  $dataset: $result_dir ($count files)" | tee -a "$LOG_FILE"
        fi
    done
    echo "" | tee -a "$LOG_FILE"
fi

if [ "$MODE" == "gpu" ] || [ "$MODE" == "all" ]; then
    echo "GPU Normal Results:" | tee -a "$LOG_FILE"
    for dataset in "${DATASETS[@]}"; do
        case "$dataset" in
            gist250k|gist500k|gist750k|gist1000k)
                size="${dataset#gist}"
                result_dir="$DATA_ROOT/gist1m/$size/results/gpu_normal"
                ;;
            video)
                result_dir="$DATA_ROOT/video/results/gpu_normal"
                ;;
            *)
                continue
                ;;
        esac
        
        if [ -d "$result_dir" ]; then
            count=$(find "$result_dir" -name "*.csv" -type f 2>/dev/null | wc -l)
            echo "  $dataset: $result_dir ($count files)" | tee -a "$LOG_FILE"
        fi
    done
    echo "" | tee -a "$LOG_FILE"
fi

if [ "$MODE" == "pq" ] || [ "$MODE" == "all" ]; then
    echo "GPU PQ Results:" | tee -a "$LOG_FILE"
    for dataset in "${DATASETS[@]}"; do
        case "$dataset" in
            gist250k|gist500k|gist750k|gist1000k)
                size="${dataset#gist}"
                result_dir="$DATA_ROOT/gist1m/$size/results/gpu_pq"
                ;;
            video)
                result_dir="$DATA_ROOT/video/results/gpu_pq"
                ;;
            *)
                continue
                ;;
        esac
        
        if [ -d "$result_dir" ]; then
            count=$(find "$result_dir" -name "*.csv" -type f 2>/dev/null | wc -l)
            echo "  $dataset: $result_dir ($count files)" | tee -a "$LOG_FILE"
        fi
    done
    echo "" | tee -a "$LOG_FILE"
fi

echo "" | tee -a "$LOG_FILE"
log_info "Detailed logs: $LOG_FILE"
echo "" | tee -a "$LOG_FILE"
