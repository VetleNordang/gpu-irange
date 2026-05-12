#!/bin/bash
#
# IDUN Build Job: iRangeGraph indexes for YouTube-8M (all 4 sizes)
#
# Builds one index per partition per modality (8 indexes total):
#   video  1m / 2m / 4m / 8m   (1024-dim RGB)
#   audi   1m / 2m / 4m / 8m   ( 128-dim audio)
#
# Skips any index that already exists.
# Submit via submit_yt8m_pipeline.sh — do not submit this directly.
#
# Submit from the repo root:
#   sbatch scripts/idun/yt8m_build_indexes.slurm
#

#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=128G
#SBATCH --job-name="yt8m_build"
#SBATCH --output=logs/yt8m_build_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/idun/log_sysinfo.sh
echo "YouTube-8M Index Build started: $(date)"
log_cpu_sysinfo

module purge
module load GCCcore/11.3.0
module load CMake/3.23.1-GCCcore-11.3.0
module load CUDA/11.8.0

# ── Build binary if needed ────────────────────────────────────────────────────
if [ ! -f "build/tests/buildindex" ]; then
    echo "Building project..."
    mkdir -p build && cd build
    cmake .. -DCMAKE_CUDA_ARCHITECTURES="70;80" \
        -DFAISS_INCLUDE_DIR=/cluster/home/vetlean/.conda/envs/irange/include \
        -DFAISS_LIBRARY=/cluster/home/vetlean/.conda/envs/irange/lib/libfaiss.so
    make -j "$SLURM_CPUS_PER_TASK"
    cd ..
else
    echo "Using existing build."
fi

BUILDINDEX="./build/tests/buildindex"
M=32
EF=200
THREADS=$SLURM_CPUS_PER_TASK

echo ""
echo "M=$M  ef_construction=$EF  threads=$THREADS"

# ── Helper ────────────────────────────────────────────────────────────────────
build_index() {
    local label="$1"
    local data="$2"
    local index="$3"

    echo ""
    echo "--- $label ---"

    if [ ! -f "$data" ]; then
        echo "SKIP: data not found at $data"
        return
    fi
    if [ -f "$index" ]; then
        echo "SKIP: index already exists at $index"
        return
    fi

    mkdir -p "$(dirname "$index")"
    echo "Building $index ..."
    $BUILDINDEX \
        --data_path  "$data"  \
        --index_file "$index" \
        --M          $M       \
        --ef_construction $EF \
        --threads    $THREADS

    if [ $? -eq 0 ]; then
        SIZE=$(du -sh "$index" | cut -f1)
        echo "Done: $index  ($SIZE)"
    else
        echo "ERROR: build failed for $label"
    fi
}

# ── Build all indexes ─────────────────────────────────────────────────────────
for SIZE in 1m 2m 4m 8m; do

    build_index "video $SIZE" \
        "executable_data/video/$SIZE/youtube_rgb_$SIZE.bin" \
        "executable_data/video/$SIZE/youtube_rgb_$SIZE.index"

    build_index "audi $SIZE" \
        "executable_data/audi/$SIZE/yt_aud_$SIZE.bin" \
        "executable_data/audi/$SIZE/yt_aud_$SIZE.index"

done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "Index build complete: $(date)"
echo ""
echo "Indexes:"
for SIZE in 1m 2m 4m 8m; do
    for MOD in video audi; do
        case "$MOD" in
            video) IDX="executable_data/video/$SIZE/youtube_rgb_$SIZE.index" ;;
            audi)  IDX="executable_data/audi/$SIZE/yt_aud_$SIZE.index" ;;
        esac
        if [ -f "$IDX" ]; then
            echo "  [OK] $IDX  ($(du -sh "$IDX" | cut -f1))"
        else
            echo "  [--] $IDX"
        fi
    done
done
echo "=========================================="
