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
    # Process all datasets
    DATASETS_TO_PROCESS=("gist1m:250k" "gist1m:500k" "gist1m:750k" "gist1m:1000k" "audi" "video")
elif [ "$1" == "gist1m" ]; then
    if [ $# -eq 1 ]; then
        # All gist1m sizes
        DATASETS_TO_PROCESS=("gist1m:250k" "gist1m:500k" "gist1m:750k" "gist1m:1000k")
    else
        # Specific size
        DATASETS_TO_PROCESS=("gist1m:$2")
    fi
elif [ "$1" == "audi" ]; then
    DATASETS_TO_PROCESS=("audi")
elif [ "$1" == "video" ]; then
    DATASETS_TO_PROCESS=("video")
else
    log_error "Unknown dataset: $1"
    echo "Usage: $0 [all|gist1m [size]|audi|video]" | tee -a "$LOG_FILE"
    exit 1
fi

# ============================================================================
# PROCESS DATASETS
# ============================================================================

for dataset_spec in "${DATASETS_TO_PROCESS[@]}"; do
    IFS=':' read -r dataset_name dataset_size <<< "$dataset_spec"
    
    # Set parameters based on dataset
    case "$dataset_name" in
        gist1m)
            vec_dim=960
            M=320
            nbits=9
            vector_file="gist_base_${dataset_size}.bin"
            dataset_path="$DATA_ROOT/gist1m/$dataset_size"
            model_prefix="gist_${dataset_size}_pq"
            display_name="gist1m/$dataset_size"
            ;;
        audi)
            vec_dim=128
            M=32
            nbits=9
            vector_file="yt_aud_sorted_vec_by_attr.bin"
            dataset_path="$DATA_ROOT/audi"
            model_prefix="audi_pq"
            display_name="audi"
            ;;
        video)
            vec_dim=1024
            M=256
            nbits=9
            vector_file="youtube_rgb_sorted.bin"
            dataset_path="$DATA_ROOT/video"
            model_prefix="video_pq"
            display_name="video"
            ;;
        *)
            log_error "Unknown dataset: $dataset_name"
            continue
            ;;
    esac
    
    # Verify dataset exists
    if [ ! -d "$dataset_path" ]; then
        log_error "Dataset not found: $display_name at $dataset_path"
        continue
    fi
    
    # Verify vector file exists
    vector_path="$dataset_path/$vector_file"
    if [ ! -f "$vector_path" ]; then
        log_error "Vector file not found for $display_name: $vector_file"
        continue
    fi
    
    # Create PQ folder
    pq_folder="$dataset_path/pq"
    mkdir -p "$pq_folder"
    
    # Set output paths
    pq_model="$pq_folder/${model_prefix}_m${M}_nb${nbits}.faiss"
    pq_codes="$pq_folder/${model_prefix}_codes_m${M}_nb${nbits}.bin"
    
    log_header "Processing: $display_name"
    log_info "Vector dimension: $vec_dim"
    log_info "M (subspaces): $M"
    log_info "nbits: $nbits (2^$nbits = $((2**nbits)) centroids per subspace)"
    log_info "Total centroids: $((M * (2**nbits)))"
    echo "" | tee -a "$LOG_FILE"
    
    log_info "Input: $vector_file"
    log_info "Output model: $(basename "$pq_model")"
    log_info "Output codes: $(basename "$pq_codes")"
    echo "" | tee -a "$LOG_FILE"
    
    # Run PQ compression
    log_info "Starting compression..."
    if "$PQ_COMPRESS" \
        --data "$vector_path" \
        --model_out "$pq_model" \
        --codes_out "$pq_codes" \
        --M "$M" \
        --nbits "$nbits" 2>&1 | tee -a "$LOG_FILE"; then
        
        log_success "Compression completed for $display_name"
        
        # Verify output files exist
        if [ -f "$pq_model" ] && [ -f "$pq_codes" ]; then
            model_size=$(du -h "$pq_model" | cut -f1)
            codes_size=$(du -h "$pq_codes" | cut -f1)
            log_info "✓ Model file: $model_size"
            log_info "✓ Codes file: $codes_size"
        fi
    else
        log_error "Compression failed for $display_name"
    fi
    
    echo "" | tee -a "$LOG_FILE"
done

# ============================================================================
# SUMMARY
# ============================================================================
log_header "PQ Compression Complete"
log_success "All datasets processed"
echo "" | tee -a "$LOG_FILE"
log_info "Log file: $LOG_FILE"
echo "" | tee -a "$LOG_FILE"
