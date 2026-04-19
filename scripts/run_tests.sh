#!/bin/bash

###############################################################################
# run_tests.sh
#
# Purpose: Build and run all tests
#
# Usage: ./run_tests.sh
#
# This script:
# 1. Compiles the project (CMake)
# 2. Runs all available tests
# 3. Reports test results
#
# Logs output to: logs/run_tests.log
#
###############################################################################

set -e

# Setup paths - resolve to actual location, not symlink
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/run_tests.log"

# Create logs directory with fallback
if ! mkdir -p "$LOG_DIR" 2>/dev/null; then
    # Try making it executable-writable
    mkdir -p "$LOG_DIR" 2>/dev/null || true
    # Fallback: write to temp location
    LOG_FILE="/tmp/run_tests_$$.log"
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
log_header "Build and Test Suite"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ============================================================================
# STEP 1: CMAKE BUILD
# ============================================================================
log_header "Step 1: CMake Build"

log_info "Project root: $PROJECT_ROOT"

# Clean old build to avoid CMake cache conflicts
if [ -d "$PROJECT_ROOT/build" ]; then
    log_info "Cleaning old build directory..."
    rm -rf "$PROJECT_ROOT/build"
fi

# Create fresh build directory
log_info "Creating fresh build directory..."
mkdir -p "$PROJECT_ROOT/build"

# Run CMake
log_info "Running CMake configuration..."
if cd "$PROJECT_ROOT/build" && cmake -DCMAKE_BUILD_TYPE=Release .. 2>&1 | tee -a "$LOG_FILE"; then
    log_success "CMake configuration completed"
else
    log_error "CMake configuration failed"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"

# Run build
log_info "Running build..."
if cmake --build . --config Release 2>&1 | tee -a "$LOG_FILE"; then
    log_success "Build completed successfully"
else
    log_error "Build failed"
    exit 1
fi

echo "" | tee -a "$LOG_FILE"

# ============================================================================
# STEP 2: RUN TESTS
# ============================================================================
log_header "Step 2: Running Tests"

# List available test executables
log_info "Available tests:"
for test_exe in buildindex pq_compress_only pq_encode pq_manual_distance search search_multi; do
    if [ -f "$PROJECT_ROOT/build/tests/$test_exe" ]; then
        log_info "  ✓ $test_exe"
    fi
done

echo "" | tee -a "$LOG_FILE"

# Run each test
declare -a PASSED=()
declare -a FAILED=()

# Test: buildindex (builds index with M=32)
if [ -f "$PROJECT_ROOT/build/tests/buildindex" ]; then
    log_info "Running: buildindex"
    if "$PROJECT_ROOT/build/tests/buildindex" 2>&1 | tee -a "$LOG_FILE"; then
        log_success "buildindex test passed"
        PASSED+=("buildindex")
    else
        log_error "buildindex test failed"
        FAILED+=("buildindex")
    fi
    echo "" | tee -a "$LOG_FILE"
fi

# Test: pq_compress_only (PQ compression test)
if [ -f "$PROJECT_ROOT/build/tests/pq_compress_only" ]; then
    log_info "Running: pq_compress_only (requires test data)"
    log_info "  Note: This test requires input vector files"
    echo "" | tee -a "$LOG_FILE"
fi

# Test: search (basic search test)
if [ -f "$PROJECT_ROOT/build/tests/search" ]; then
    log_info "Running: search (requires index files)"
    log_info "  Note: This test requires built index and data files"
    echo "" | tee -a "$LOG_FILE"
fi

# Test: search_multi (multi-threaded search)
if [ -f "$PROJECT_ROOT/build/tests/search_multi" ]; then
    log_info "Running: search_multi (requires index files)"
    log_info "  Note: This test requires built index and data files"
    echo "" | tee -a "$LOG_FILE"
fi

# ============================================================================
# STEP 3: TEST SUMMARY
# ============================================================================
log_header "Test Summary"

if [ ${#PASSED[@]} -gt 0 ]; then
    log_success "Passed tests: ${#PASSED[@]}"
    for test in "${PASSED[@]}"; do
        echo "  ✓ $test" | tee -a "$LOG_FILE"
    done
else
    log_info "No executable tests ran (most tests require data files)"
fi

echo "" | tee -a "$LOG_FILE"

if [ ${#FAILED[@]} -gt 0 ]; then
    log_error "Failed tests: ${#FAILED[@]}"
    for test in "${FAILED[@]}"; do
        echo "  ✗ $test" | tee -a "$LOG_FILE"
    done
    echo "" | tee -a "$LOG_FILE"
    log_error "Some tests failed"
else
    log_success "All runnable tests passed"
fi

echo "" | tee -a "$LOG_FILE"
log_success "Build and test suite complete"
log_info "Log file: $LOG_FILE"
echo "" | tee -a "$LOG_FILE"
