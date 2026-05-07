#!/bin/bash
#
# check_setup.sh
#
# Validates that all expected data files, indexes, directories, and binaries
# are in place. Run on both local and IDUN to confirm setup is correct.
#
# Usage: bash scripts/check_setup.sh
#

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA="$ROOT/executable_data"

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0

ok()   { echo -e "  ${GREEN}OK${NC}    $1"; PASS=$((PASS+1)); }
fail() { echo -e "  ${RED}MISS${NC}  $1"; FAIL=$((FAIL+1)); }
warn() { echo -e "  ${YELLOW}WARN${NC}  $1"; WARN=$((WARN+1)); }

check_file() { [ -f "$1" ] && ok "$1" || fail "$1"; }
check_dir()  { [ -d "$1" ] && ok "$1" || fail "$1"; }

echo "========================================"
echo "iRangeGraph Setup Check"
echo "Host: $(hostname)"
echo "Date: $(date)"
echo "Root: $ROOT"
echo "========================================"

# ── Binaries ─────────────────────────────────────────────────────────────────
echo ""
echo "── Binaries ──"
check_file "$ROOT/build/tests/search"
check_file "$ROOT/cude_version/build/optimized_test"

# ── GIST1M ───────────────────────────────────────────────────────────────────
echo ""
echo "── GIST1M ──"
for size in 250k 500k 750k 1000k; do
    echo " [$size]"
    dir="$DATA/gist1m/$size"
    check_file "$dir/gist_base_${size}.bin"
    check_file "$dir/gist_query_${size}.bin"
    check_file "$dir/gist_attr_${size}.bin"
    check_file "$dir/gist_${size}.index"
    check_dir  "$dir/groundtruth"
    check_dir  "$dir/query_ranges"
    check_dir  "$dir/results"
done

# ── VIDEO ─────────────────────────────────────────────────────────────────────
echo ""
echo "── Video (YouTube RGB) ──"
for size in 1m 2m 4m 8m; do
    echo " [$size]"
    dir="$DATA/video/$size"
    check_file "$dir/youtube_rgb_${size}.bin"
    check_file "$dir/youtube_rgb_query.bin"
    check_file "$dir/youtube_rgb_attr_${size}.bin"
    check_file "$dir/youtube_rgb_${size}.index"
    check_dir  "$dir/groundtruth"
    check_dir  "$dir/query_ranges"
    check_dir  "$dir/results"
done

# ── AUDI ──────────────────────────────────────────────────────────────────────
echo ""
echo "── Audi (YouTube Audio) ──"
for size in 1m 2m 4m 8m; do
    echo " [$size]"
    dir="$DATA/audi/$size"
    check_file "$dir/yt_aud_${size}.bin"
    check_file "$dir/yt_aud_query.bin"
    check_file "$dir/yt_aud_attr_${size}.bin"
    check_file "$dir/yt_aud_${size}.index"
    check_dir  "$dir/groundtruth"
    check_dir  "$dir/query_ranges"
    check_dir  "$dir/results"
done

# ── Results check (warn if empty) ─────────────────────────────────────────────
echo ""
echo "── Result CSVs ──"
for size in 250k 500k 750k 1000k; do
    count=$(find "$DATA/gist1m/$size/results" -name "*.csv" 2>/dev/null | wc -l)
    [ "$count" -gt 0 ] && ok "gist1m/$size/results ($count CSVs)" || warn "gist1m/$size/results (no CSVs yet)"
done
for mod in video audi; do
    for size in 1m 2m 4m 8m; do
        count=$(find "$DATA/$mod/$size/results" -name "*.csv" 2>/dev/null | wc -l)
        [ "$count" -gt 0 ] && ok "$mod/$size/results ($count CSVs)" || warn "$mod/$size/results (no CSVs yet)"
    done
done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo -e "  ${GREEN}OK${NC}:      $PASS"
echo -e "  ${YELLOW}WARN${NC}:    $WARN  (missing CSVs — data not yet generated)"
echo -e "  ${RED}MISSING${NC}: $FAIL"
echo "========================================"
if [ "$FAIL" -eq 0 ]; then
    echo -e "${GREEN}All required files present. Ready to run.${NC}"
else
    echo -e "${RED}$FAIL required files/dirs missing. Fix before running jobs.${NC}"
fi
echo ""
