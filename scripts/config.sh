#!/bin/bash
# Single source of truth for all paths and constants.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/config.sh"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_ROOT="$PROJECT_ROOT/executable_data"

# Result subdirectory names — change here, change nowhere else
DIR_CPU_SERIAL="cpu_serial"
DIR_CPU_PARALLEL="cpu_parallel"
DIR_GPU_NORMAL="gpu_normal"
DIR_GPU_PQ="gpu_pq"
DIR_ANALYSIS="analysis"

GRAPH_M=32

CPU_SEARCH_BIN="$PROJECT_ROOT/build/tests/search"
GPU_SEARCH_BIN="$PROJECT_ROOT/cude_version/build/optimized_test"
GPU_PQ_BIN="$PROJECT_ROOT/cude_version/build/hello_pq"
