#!/bin/bash
#
# download_data.sh
#
# Downloads and converts datasets into iRangeGraph binary format.
# Does NOT build any index — run execute_experiments.sh after this.
#
# Usage:
#   ./download_data.sh            # all datasets
#   ./download_data.sh gist       # GIST1M only
#   ./download_data.sh youtube    # YouTube-8M (video + audio) only
#   ./download_data.sh gist youtube   # explicit list
#
# Sizes produced:
#   GIST1M  : 250k / 500k / 750k / 1000k  (exectable_data/gist1m/)
#   YouTube : 1M                           (exectable_data/video/1m/ and audi/1m/)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/download_data.log"
PYTHON_DIR="$PROJECT_ROOT/python"
DATA_ROOT="$PROJECT_ROOT/exectable_data"

mkdir -p "$LOG_DIR"
> "$LOG_FILE"

# ── colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; NC='\033[0m'

log_header()  { echo -e "${BLUE}╔════════════════════════════════════════╗${NC}" | tee -a "$LOG_FILE"
                echo -e "${BLUE}║${NC} $1" | tee -a "$LOG_FILE"
                echo -e "${BLUE}╚════════════════════════════════════════╝${NC}" | tee -a "$LOG_FILE"; }
log_success() { echo -e "${GREEN}✓${NC} $1" | tee -a "$LOG_FILE"; }
log_info()    { echo -e "${YELLOW}ℹ${NC} $1" | tee -a "$LOG_FILE"; }
log_error()   { echo -e "${RED}✗${NC} $1" | tee -a "$LOG_FILE"; }

# ── resolve which python to use ───────────────────────────────────────────────
resolve_python() {
    # prefer conda env, then venv, then system
    CONDA_PY="/cluster/home/vetlean/.conda/envs/irange/bin/python3"
    if [ -f "$CONDA_PY" ]; then
        echo "$CONDA_PY"; return
    fi
    if [ -n "$CONDA_PREFIX" ] && [ -f "$CONDA_PREFIX/bin/python3" ]; then
        echo "$CONDA_PREFIX/bin/python3"; return
    fi
    echo "python3"
}

PYTHON="$(resolve_python)"
log_info "Using Python: $PYTHON"

# ── parse arguments ───────────────────────────────────────────────────────────
TARGETS=("$@")
if [ ${#TARGETS[@]} -eq 0 ]; then
    TARGETS=("gist" "youtube")
fi

DO_GIST=0; DO_YOUTUBE=0
for t in "${TARGETS[@]}"; do
    case "$t" in
        gist)    DO_GIST=1 ;;
        youtube|video|audi) DO_YOUTUBE=1 ;;
        all)     DO_GIST=1; DO_YOUTUBE=1 ;;
        *) log_error "Unknown target '$t'. Valid: gist youtube all"; exit 1 ;;
    esac
done

echo "" | tee -a "$LOG_FILE"
log_header "Dataset Download and Preparation"
echo "Timestamp: $(date)" | tee -a "$LOG_FILE"
echo "Log file : $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# ══════════════════════════════════════════════════════════════════════════════
# GIST1M
# ══════════════════════════════════════════════════════════════════════════════
if [ "$DO_GIST" -eq 1 ]; then
    log_header "GIST1M (~3.6 GB download)"

    GIST_RAW="$DATA_ROOT/gist1m/gist"
    mkdir -p "$GIST_RAW"

    # ── Step 1: download ──────────────────────────────────────────────────────
    if [ -f "$GIST_RAW/gist_base.fvecs" ] && [ -f "$GIST_RAW/gist_query.fvecs" ]; then
        log_success "Raw .fvecs files already present — skipping download"
    else
        log_info "Downloading from ftp.irisa.fr ..."
        wget -c --progress=bar:force \
             -O "$GIST_RAW/gist.tar.gz" \
             ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz \
             2>&1 | tee -a "$LOG_FILE"

        log_info "Extracting archive ..."
        tar -xzf "$GIST_RAW/gist.tar.gz" -C "$GIST_RAW" --strip-components=1
        rm "$GIST_RAW/gist.tar.gz"
        log_success "Download and extraction complete"
    fi

    echo "Raw files:" | tee -a "$LOG_FILE"
    ls -lh "$GIST_RAW" | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"

    # ── Step 2: convert to iRangeGraph .bin ──────────────────────────────────
    CONVERT_SCRIPT="$PYTHON_DIR/gist/convert_gist_to_irangegraph.py"
    if [ ! -f "$CONVERT_SCRIPT" ]; then
        log_error "Conversion script not found: $CONVERT_SCRIPT"
        exit 1
    fi

    # Check if all output sizes already exist
    ALL_SIZES_EXIST=1
    for size in 250k 500k 750k 1000k; do
        if [ ! -f "$DATA_ROOT/gist1m/$size/gist_base_${size}.bin" ]; then
            ALL_SIZES_EXIST=0; break
        fi
    done

    if [ "$ALL_SIZES_EXIST" -eq 1 ]; then
        log_success "All GIST binary files already exist — skipping conversion"
    else
        log_info "Converting .fvecs → iRangeGraph .bin (250k / 500k / 750k / 1000k) ..."
        "$PYTHON" "$CONVERT_SCRIPT" \
            --base_fvecs  "$GIST_RAW/gist_base.fvecs" \
            --query_fvecs "$GIST_RAW/gist_query.fvecs" \
            --out_root    "$DATA_ROOT/gist1m" \
            --sizes       250000,500000,750000,1000000 \
            2>&1 | tee -a "$LOG_FILE"
        log_success "GIST1M conversion complete"
    fi

    echo "" | tee -a "$LOG_FILE"
    log_info "Output files:"
    for size in 250k 500k 750k 1000k; do
        DIR="$DATA_ROOT/gist1m/$size"
        if [ -d "$DIR" ]; then
            echo "  $size:" | tee -a "$LOG_FILE"
            ls -lh "$DIR"/*.bin 2>/dev/null | awk '{print "    "$NF, $5}' | tee -a "$LOG_FILE"
        fi
    done
    echo "" | tee -a "$LOG_FILE"
fi

# ══════════════════════════════════════════════════════════════════════════════
# YouTube-8M  (video RGB + audio)
# ══════════════════════════════════════════════════════════════════════════════
if [ "$DO_YOUTUBE" -eq 1 ]; then
    log_header "YouTube-8M video+audio (~68 GB download, 1M partition)"

    CONVERT_SCRIPT="$PYTHON_DIR/youtube8m/convert_youtube8m_to_irangegraph.py"
    CHECK_SCRIPT="$PYTHON_DIR/youtube8m/check_yt8m_bins.py"

    if [ ! -f "$CONVERT_SCRIPT" ]; then
        log_error "Conversion script not found: $CONVERT_SCRIPT"
        exit 1
    fi

    # ── Step 1: Python dependencies ───────────────────────────────────────────
    log_info "Checking Python dependencies (tensorflow, numpy) ..."
    if ! "$PYTHON" -c "import tensorflow" 2>/dev/null; then
        log_info "Installing tensorflow-cpu (this may take a few minutes) ..."
        "$PYTHON" -m pip install --quiet "tensorflow-cpu>=2.10" numpy
    fi
    log_success "Python dependencies OK"

    # ── Step 2: download + convert ────────────────────────────────────────────
    # Use /tmp or $TMPDIR for raw TFRecords (large, deleted after conversion)
    RAW_DIR="${TMPDIR:-/tmp}/yt8m_raw_$$"
    mkdir -p "$RAW_DIR"
    log_info "Temporary TFRecord storage: $RAW_DIR  (deleted after conversion)"

    # Check if 1M output already exists for both modalities
    if [ -f "$DATA_ROOT/video/1m/youtube_rgb_1m.bin" ] && \
       [ -f "$DATA_ROOT/audi/1m/yt_aud_1m.bin" ]; then
        log_success "YouTube-8M 1M binary files already exist — skipping download"
    else
        log_info "Downloading and converting YouTube-8M (1M partition) ..."
        log_info "This downloads ~68 GB and may take several hours."
        echo "" | tee -a "$LOG_FILE"

        "$PYTHON" "$CONVERT_SCRIPT" \
            --raw_dir     "$RAW_DIR" \
            --out_root    "$DATA_ROOT" \
            --sizes       "1000000" \
            --num_queries 1000 \
            --chunk_size  500000 \
            --mirror      us \
            2>&1 | tee -a "$LOG_FILE"

        log_success "YouTube-8M conversion complete"
        rm -rf "$RAW_DIR"
        log_info "Removed temporary TFRecord files"
    fi

    # ── Step 3: verify ────────────────────────────────────────────────────────
    if [ -f "$CHECK_SCRIPT" ]; then
        log_info "Verifying output files ..."
        "$PYTHON" "$CHECK_SCRIPT" 2>&1 | tee -a "$LOG_FILE"
    fi

    echo "" | tee -a "$LOG_FILE"
    log_info "Output files:"
    for modality in video audi; do
        DIR="$DATA_ROOT/$modality/1m"
        if [ -d "$DIR" ]; then
            echo "  $modality/1m:" | tee -a "$LOG_FILE"
            ls -lh "$DIR"/*.bin 2>/dev/null | awk '{print "    "$NF, $5}' | tee -a "$LOG_FILE"
        fi
    done
    echo "" | tee -a "$LOG_FILE"
fi

# ══════════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════════
log_header "Download complete"
log_success "All requested datasets are ready"
echo "" | tee -a "$LOG_FILE"
log_info "Next steps:"
echo "  cd ~/irange/scripts" | tee -a "$LOG_FILE"
echo "  ./setup_folders.sh" | tee -a "$LOG_FILE"
echo "  ./make_pq.sh all" | tee -a "$LOG_FILE"
echo "  ./execute_experiments.sh all" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
log_info "Log saved to: $LOG_FILE"
