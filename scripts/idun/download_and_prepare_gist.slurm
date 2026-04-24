#!/bin/bash
#
# IDUN Data Preparation Job: Download and convert GIST1M dataset
# Run once before any index build or GPU search jobs.
#
# Submit from the repo root:
#   sbatch download_and_prepare_gist.slurm
#

#SBATCH --partition=CPUQ
#SBATCH --account=vetlean      # TODO: replace (run: sacctmgr show assoc)
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name="gist_prepare"
#SBATCH --output=logs/gist_prepare_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "GIST1M preparation started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "=========================================="

REPO_DIR="$SLURM_SUBMIT_DIR"
GIST_RAW_DIR="$REPO_DIR/exectable_data/gist1m/gist"

mkdir -p "$GIST_RAW_DIR"
mkdir -p "$REPO_DIR/logs"

# -------------------------------------------------------
# Step 1: Download
# -------------------------------------------------------
echo ""
echo "--- Step 1: Downloading GIST1M (~3.6 GB) ---"

if [ -f "$GIST_RAW_DIR/gist_base.fvecs" ] && [ -f "$GIST_RAW_DIR/gist_query.fvecs" ]; then
    echo "Raw .fvecs files already present, skipping download."
else
    wget -c -O "$GIST_RAW_DIR/gist.tar.gz" \
        ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz

    if [ $? -ne 0 ]; then
        echo "ERROR: Download failed."
        exit 1
    fi

    echo "Extracting..."
    tar -xzf "$GIST_RAW_DIR/gist.tar.gz" -C "$GIST_RAW_DIR" --strip-components=1
    rm "$GIST_RAW_DIR/gist.tar.gz"
    echo "Done."
fi

echo "Raw files:"
ls -lh "$GIST_RAW_DIR"

# -------------------------------------------------------
# Step 2: Convert .fvecs -> iRangeGraph .bin
# -------------------------------------------------------
echo ""
echo "--- Step 2: Converting to iRangeGraph binary format ---"

module purge
module load Python/3.10.8-GCCcore-12.2.0

python3 -c "import numpy" 2>/dev/null || pip install --user numpy

python3 "$REPO_DIR/python/gist/convert_gist_to_irangegraph.py" \
    --base_fvecs  "$GIST_RAW_DIR/gist_base.fvecs" \
    --query_fvecs "$GIST_RAW_DIR/gist_query.fvecs" \
    --out_root    "$REPO_DIR/exectable_data/gist1m" \
    --sizes       250000,500000,750000,1000000

if [ $? -ne 0 ]; then
    echo "ERROR: Conversion script failed."
    exit 1
fi

# -------------------------------------------------------
# Summary
# -------------------------------------------------------
echo ""
echo "--- Generated files ---"
for size in 250k 500k 750k 1000k; do
    DIR="$REPO_DIR/exectable_data/gist1m/$size"
    if [ -d "$DIR" ]; then
        echo "$size:"
        ls -lh "$DIR"/*.bin 2>/dev/null
    fi
done

echo ""
echo "=========================================="
echo "Data preparation complete: $(date)"
echo "Next step: sbatch scripts/idun/gpu_job.slurm"
echo "=========================================="
