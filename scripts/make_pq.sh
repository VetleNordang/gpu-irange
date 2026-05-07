#!/bin/bash

###############################################################################
# make_pq.sh
#
# Purpose: Create PQ (Product Quantization) compressed versions of datasets
#
# Usage: ./make_pq.sh [all|dataset [size]]
#        ./make_pq.sh all              # Compress all datasets
#        ./make_pq.sh gist1m 250k      # Compress only gist1m 250k
#        ./make_pq.sh audi             # Compress audi dataset
#
# Parameters for each dataset:
#   - gist1m (384-dim):   M=320, nbits=9    → 320 subspaces × 512 centroids
#   - audi (128-dim):     M=32,  nbits=9    → 32 subspaces × 512 centroids
#   - video (1024-dim):   M=256, nbits=9    → 256 subspaces × 512 centroids
#
# Output files:
#   - .faiss files (PQ model)
#   - .bin files (encoded PQ codes)
#
# Logs output to: logs/make_pq.log
#
###############################################################################

set -e

# Setup paths - resolve to actual location, not symlink
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_ROOT="$PROJECT_ROOT/executable_data"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/make_pq.log"
PQ_COMPRESS="$PROJECT_ROOT/build/tests/pq_compress_only"

# Create logs directory with fallback
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    # Try making it executable-writable
    mkdir -p "$LOG_DIR" 2>/dev/null || true
    # Fallback: write to temp location
    LOG_FILE="/tmp/make_pq_$$.log"
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
log_header "PQ Dataset Compression"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Verify pq_compress_only exists
if [ ! -f "$PQ_COMPRESS" ]; then
    log_error "pq_compress_only executable not found: $PQ_COMPRESS"
    log_info "Please compile first: cd $PROJECT_ROOT && cmake -B build && cmake --build build"
    exit 1
fi

log_success "Found pq_compress_only executable"
echo "" | tee -a "$LOG_FILE"

# Parse arguments
DATASETS_TO_PROCESS=()

if [ $# -eq 0 ] || [ "$1" == "all" ]; then
    DATASETS_TO_PROCESS=(
        "gist1m:250k" "gist1m:500k" "gist1m:750k" "gist1m:1000k"
        "video:1m" "video:2m" "video:4m" "video:8m"
        "audi:1m"  "audi:2m"  "audi:4m"  "audi:8m"
    )
elif [ "$1" == "gist1m" ]; then
    if [ $# -eq 1 ]; then
        DATASETS_TO_PROCESS=("gist1m:250k" "gist1m:500k" "gist1m:750k" "gist1m:1000k")
    else
        DATASETS_TO_PROCESS=("gist1m:$2")
    fi
elif [ "$1" == "video" ]; then
    if [ $# -eq 1 ]; then
        DATASETS_TO_PROCESS=("video:1m" "video:2m" "video:4m" "video:8m")
    else
        DATASETS_TO_PROCESS=("video:$2")
    fi
elif [ "$1" == "audi" ]; then
    if [ $# -eq 1 ]; then
        DATASETS_TO_PROCESS=("audi:1m" "audi:2m" "audi:4m" "audi:8m")
    else
        DATASETS_TO_PROCESS=("audi:$2")
    fi
else
    log_error "Unknown dataset: $1"
    echo "Usage: $0 [all|gist1m [size]|video [size]|audi [size]]" | tee -a "$LOG_FILE"
    exit 1
fi

# ============================================================================
# PROCESS DATASETS
# ============================================================================

compress_dataset() {
    local dataset_name="$1"
    local dataset_size="$2"
    local vec_dim M nbits vector_file dataset_path model_prefix display_name

    case "$dataset_name" in
        gist1m)
            vec_dim=960; M=320; nbits=9
            vector_file="gist_base_${dataset_size}.bin"
            dataset_path="$DATA_ROOT/gist1m/$dataset_size"
            model_prefix="gist_${dataset_size}_pq"
            display_name="gist1m/$dataset_size"
            ;;
        video)
            vec_dim=1024; M=256; nbits=9
            vector_file="youtube_rgb_${dataset_size}.bin"
            dataset_path="$DATA_ROOT/video/$dataset_size"
            model_prefix="video_${dataset_size}_pq"
            display_name="video/$dataset_size"
            ;;
        audi)
            vec_dim=128; M=32; nbits=9
            vector_file="yt_aud_${dataset_size}.bin"
            dataset_path="$DATA_ROOT/audi/$dataset_size"
            model_prefix="audi_${dataset_size}_pq"
            display_name="audi/$dataset_size"
            ;;
        *)
            log_error "Unknown dataset: $dataset_name"
            return 1
            ;;
    esac

    local vector_path="$dataset_path/$vector_file"
    local pq_folder="$dataset_path/pq"
    local pq_model="$pq_folder/${model_prefix}_m${M}_nb${nbits}.faiss"
    local pq_codes="$pq_folder/${model_prefix}_codes_m${M}_nb${nbits}.bin"

    log_header "Processing: $display_name"

    if [ ! -f "$vector_path" ]; then
        log_error "Vector file not found: $vector_path — skipping"
        return 0
    fi

    if [ -f "$pq_model" ] && [ -f "$pq_codes" ]; then
        log_info "Already trained — skipping (delete files to retrain)"
        log_info "  $pq_model"
        log_info "  $pq_codes"
        echo "" | tee -a "$LOG_FILE"
        return 0
    fi

    mkdir -p "$pq_folder"

    log_info "dim=$vec_dim  M=$M  nbits=$nbits  ($(( M * (2**nbits) )) total centroids)"
    log_info "Input:  $vector_path"
    log_info "Output: $(basename "$pq_model"), $(basename "$pq_codes")"
    echo "" | tee -a "$LOG_FILE"

    if "$PQ_COMPRESS" \
        --data "$vector_path" \
        --model_out "$pq_model" \
        --codes_out "$pq_codes" \
        --M "$M" \
        --nbits "$nbits" 2>&1 | tee -a "$LOG_FILE"; then

        if [ -f "$pq_model" ] && [ -f "$pq_codes" ]; then
            log_success "Done: $(du -h "$pq_model" | cut -f1) model, $(du -h "$pq_codes" | cut -f1) codes"
        else
            log_error "Binary exited 0 but output files missing"
        fi
    else
        log_error "Compression failed for $display_name"
    fi

    echo "" | tee -a "$LOG_FILE"
}

for dataset_spec in "${DATASETS_TO_PROCESS[@]}"; do
    IFS=':' read -r dataset_name dataset_size <<< "$dataset_spec"
    compress_dataset "$dataset_name" "$dataset_size"
done

# ============================================================================
# SUMMARY
# ============================================================================
log_header "PQ Compression Complete"
log_success "All datasets processed"
echo "" | tee -a "$LOG_FILE"
log_info "Log file: $LOG_FILE"
echo "" | tee -a "$LOG_FILE"
