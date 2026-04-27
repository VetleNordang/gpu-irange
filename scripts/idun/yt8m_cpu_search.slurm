#!/bin/bash
#
# IDUN CPU Search Job: iRangeGraph search on all YouTube-8M indexes
#
# Runs search on all 8 index/dataset pairs (1m, 2m, 4m, 8m × video + audi).
# Skips any pair whose index does not exist yet.
#
# Normally submitted automatically by submit_yt8m_pipeline.sh after the
# build job completes. Can also be run standalone if indexes are already built:
#   sbatch scripts/idun/yt8m_cpu_search.slurm
#

#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name="yt8m_search"
#SBATCH --output=logs/yt8m_search_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "YouTube-8M CPU search started: $(date)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "Cores:   $SLURM_CPUS_PER_TASK"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"

module purge
module load GCCcore/11.3.0
module load CUDA/11.8.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

SEARCH="./build/tests/search"
M=32

if [ ! -f "$SEARCH" ]; then
    echo "ERROR: $SEARCH not found. Run yt8m_build_indexes.slurm first."
    exit 1
fi

# ── Helper ────────────────────────────────────────────────────────────────────
run_search() {
    local label="$1"
    local data="$2"
    local query="$3"
    local index="$4"
    local range_prefix="$5"
    local gt_prefix="$6"
    local result_prefix="$7"

    echo ""
    echo "--- $label ---"

    if [ ! -f "$index" ]; then
        echo "SKIP: index not found at $index"
        return
    fi

    mkdir -p "$(dirname "$range_prefix")"
    mkdir -p "$(dirname "$gt_prefix")"
    mkdir -p "$(dirname "$result_prefix")"

    $SEARCH \
        --data_path              "$data"          \
        --query_path             "$query"         \
        --index_file             "$index"         \
        --range_saveprefix       "$range_prefix"  \
        --groundtruth_saveprefix "$gt_prefix"     \
        --result_saveprefix      "$result_prefix" \
        --M $M

    if [ $? -eq 0 ]; then
        echo "Done: results at ${result_prefix}*.csv"
    else
        echo "ERROR: search failed for $label"
    fi
}

# ── Run all searches ──────────────────────────────────────────────────────────
for SIZE in 1m 2m 4m 8m; do

    run_search "video $SIZE" \
        "exectable_data/video/$SIZE/youtube_rgb_$SIZE.bin"   \
        "exectable_data/video/$SIZE/youtube_rgb_query.bin"   \
        "exectable_data/video/$SIZE/youtube_rgb_$SIZE.index" \
        "exectable_data/video/$SIZE/query_ranges/qr"         \
        "exectable_data/video/$SIZE/groundtruth/gt"          \
        "exectable_data/video/$SIZE/results/res"

    run_search "audi $SIZE" \
        "exectable_data/audi/$SIZE/yt_aud_$SIZE.bin"   \
        "exectable_data/audi/$SIZE/yt_aud_query.bin"   \
        "exectable_data/audi/$SIZE/yt_aud_$SIZE.index" \
        "exectable_data/audi/$SIZE/query_ranges/qr"    \
        "exectable_data/audi/$SIZE/groundtruth/gt"     \
        "exectable_data/audi/$SIZE/results/res"

done

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "CPU search complete: $(date)"
echo ""
echo "Results:"
for SIZE in 1m 2m 4m 8m; do
    for MOD in video audi; do
        DIR="exectable_data/$MOD/$SIZE/results"
        COUNT=$(ls "$DIR"/*.csv 2>/dev/null | wc -l)
        if [ "$COUNT" -gt 0 ]; then
            echo "  [OK] $MOD/$SIZE  ($COUNT CSV files)"
        else
            echo "  [--] $MOD/$SIZE  (no results)"
        fi
    done
done
echo "=========================================="
