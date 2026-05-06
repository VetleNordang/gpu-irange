#!/bin/bash
#
# IDUN Data Preparation Job: Download and convert YouTube-8M (all 4 sizes)
#
# Produces 1M, 2M, 4M, and 8M vector partitions for both:
#   video-RGB  (1024-dim)  →  executable_data/video/{1m,2m,4m,8m}/
#   audio      ( 128-dim)  →  executable_data/audi/{1m,2m,4m,8m}/
#
# Disk usage:
#   Scratch ($USERWORK) during job : ~68 GB  (TFRecords + temp, auto-deleted)
#   Permanent output in repo       : ~69 GB  (final .bin files kept)
#
# Submit from the repo root:
#   sbatch scripts/idun/download_and_prepare_youtube8m.slurm
#

#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name="yt8m_prepare"
#SBATCH --output=logs/yt8m_prepare_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "YouTube-8M preparation started: $(date)"
echo "Job ID:  $SLURM_JOB_ID"
echo "Node:    $(hostname)"
echo "=========================================="

REPO_DIR="$SLURM_SUBMIT_DIR"
mkdir -p "$REPO_DIR/logs"

# Raw TFRecords and temp memmaps go to scratch ($USERWORK) so they don't
# eat into the home-directory quota. They are deleted by the script when done.
# If $USERWORK is not set on this cluster, fall back to the repo directory.
if [ -n "$USERWORK" ]; then
    RAW_DIR="$USERWORK/yt8m_raw_${SLURM_JOB_ID}"
else
    RAW_DIR="$REPO_DIR/executable_data/yt8m_raw"
fi

echo "Scratch dir  : $RAW_DIR"
echo "Output dir   : $REPO_DIR/executable_data"
echo ""

# ── Step 1: Set up Python environment ────────────────────────────────────────
echo "--- Step 1: Python environment ---"

module purge
module load Python/3.10.8-GCCcore-12.2.0

VENV_DIR="$REPO_DIR/.venv_yt8m"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR ..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet "tensorflow-cpu>=2.10" numpy

python3 -c "import tensorflow as tf; print('TensorFlow', tf.__version__)"
if [ $? -ne 0 ]; then
    echo "ERROR: TensorFlow import failed."
    exit 1
fi

# ── Step 2: Download + convert ────────────────────────────────────────────────
#
# The Python script handles everything in one go:
#   - Downloads training shards  (base vectors, auto-calculates how many)
#   - Downloads validation shard (query vectors, non-overlapping with train)
#   - Extracts RGB + audio features per frame, dequantizes to float32
#   - Sorts by L2-norm attribute
#   - Writes all 4 size partitions
#   - Deletes raw TFRecords and temp files when done
#
echo ""
echo "--- Step 2: Download and convert (1M, 2M, 4M, 8M) ---"

python3 "$REPO_DIR/python/youtube8m/convert_youtube8m_to_irangegraph.py" \
    --raw_dir     "$RAW_DIR"                        \
    --out_root    "$REPO_DIR/executable_data"         \
    --sizes       "1000000,2000000,4000000,8000000"  \
    --num_queries 1000                               \
    --chunk_size  500000                             \
    --mirror      us

if [ $? -ne 0 ]; then
    echo "ERROR: Conversion failed."
    exit 1
fi

# ── Step 3: Verify output ─────────────────────────────────────────────────────
echo ""
echo "--- Step 3: Verifying output files ---"

python3 "$REPO_DIR/python/youtube8m/check_yt8m_bins.py"

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "--- Generated files ---"
for modality in video audi; do
    for size in 1m 2m 4m 8m; do
        DIR="$REPO_DIR/executable_data/$modality/$size"
        if [ -d "$DIR" ]; then
            echo "$modality/$size:"
            ls -lh "$DIR"/*.bin 2>/dev/null | awk '{print "  "$NF, $5}'
        fi
    done
done

echo ""
echo "=========================================="
echo "YouTube-8M preparation complete: $(date)"
echo ""
echo "Next: build an index, e.g.:"
echo "  sbatch scripts/idun/gpu_job.slurm"
echo "=========================================="
