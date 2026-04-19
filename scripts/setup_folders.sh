#!/bin/bash

###############################################################################
# setup_folders.sh
#
# Purpose: Create organized folder structure for experiments
#
# Usage: ./setup_folders.sh
#
# Creates:
# - exectable_data/gist1m/[250k|500k|750k|1000k]/results/{cpu,gpu_normal,gpu_pq,analysis}/
# - exectable_data/audi/results/{cpu,gpu_normal,gpu_pq,analysis}/
# - exectable_data/video/results/{cpu,gpu_normal,gpu_pq,analysis}/
#
# Logs output to: logs/setup_folders.log
#
###############################################################################

set -e

# Setup paths - resolve to actual location, not symlink
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
DATA_ROOT="$PROJECT_ROOT/exectable_data"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/setup_folders.log"

# Create logs directory with fallback
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    # Try making it executable-writable
    mkdir -p "$LOG_DIR" 2>/dev/null || true
    # Fallback: write to temp location
    LOG_FILE="/tmp/setup_folders_$$.log"
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
log_header "Setup Folder Structure for Experiments"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Verify data root exists
if [ ! -d "$DATA_ROOT" ]; then
    log_error "Data root not found: $DATA_ROOT"
    exit 1
fi

log_info "Data root: $DATA_ROOT"
echo "" | tee -a "$LOG_FILE"

# Datasets configuration
declare -a GIST_SIZES=("250k" "500k" "750k" "1000k")
declare -a RESULT_SUBDIRS=("cpu" "gpu_normal" "gpu_pq" "analysis")

# ============================================================================
# GIST1M DATASETS (with size variations)
# ============================================================================
log_header "Creating GIST1M Folders"

for size in "${GIST_SIZES[@]}"; do
    dataset_dir="$DATA_ROOT/gist1m/$size"
    results_dir="$dataset_dir/results"
    
    if [ ! -d "$dataset_dir" ]; then
        log_error "Dataset not found: $dataset_dir"
        continue
    fi
    
    # Create results directory
    mkdir -p "$results_dir"
    log_info "Created: gist1m/$size/results"
    
    # Create subdirectories
    for subdir in "${RESULT_SUBDIRS[@]}"; do
        target_dir="$results_dir/$subdir"
        mkdir -p "$target_dir"
        touch "$target_dir/.gitkeep"
        log_info "  ├─ Created: $subdir/"
    done
done

echo "" | tee -a "$LOG_FILE"

# ============================================================================
# AUDI DATASET
# ============================================================================
log_header "Creating AUDI Folders"

audi_dir="$DATA_ROOT/audi"
if [ -d "$audi_dir" ]; then
    audi_results="$audi_dir/results"
    mkdir -p "$audi_results"
    log_info "Created: audi/results"
    
    for subdir in "${RESULT_SUBDIRS[@]}"; do
        target_dir="$audi_results/$subdir"
        mkdir -p "$target_dir"
        touch "$target_dir/.gitkeep"
        log_info "  ├─ Created: $subdir/"
    done
else
    log_error "Audi dataset not found: $audi_dir"
fi

echo "" | tee -a "$LOG_FILE"

# ============================================================================
# VIDEO DATASET
# ============================================================================
log_header "Creating VIDEO Folders"

video_dir="$DATA_ROOT/video"
if [ -d "$video_dir" ]; then
    video_results="$video_dir/results"
    mkdir -p "$video_results"
    log_info "Created: video/results"
    
    for subdir in "${RESULT_SUBDIRS[@]}"; do
        target_dir="$video_results/$subdir"
        mkdir -p "$target_dir"
        touch "$target_dir/.gitkeep"
        log_info "  ├─ Created: $subdir/"
    done
else
    log_error "Video dataset not found: $video_dir"
fi

echo "" | tee -a "$LOG_FILE"

# ============================================================================
# SUMMARY
# ============================================================================
log_header "Setup Complete"
log_success "All folders created successfully"
echo "" | tee -a "$LOG_FILE"
log_info "Log file: $LOG_FILE"
echo "" | tee -a "$LOG_FILE"

echo "Folder structure created:" | tee -a "$LOG_FILE"
echo "  ✓ exectable_data/gist1m/{250k,500k,750k,1000k}/results/{cpu,gpu_normal,gpu_pq,analysis}/" | tee -a "$LOG_FILE"
echo "  ✓ exectable_data/audi/results/{cpu,gpu_normal,gpu_pq,analysis}/" | tee -a "$LOG_FILE"
echo "  ✓ exectable_data/video/results/{cpu,gpu_normal,gpu_pq,analysis}/" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
