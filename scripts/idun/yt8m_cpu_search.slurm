#!/bin/bash
#
# IDUN CPU Search Job: iRangeGraph search on all YouTube-8M indexes
#
# Runs serial then parallel on all 8 index/dataset pairs (1m–8m × video + audi).
# Skips any pair whose index does not exist yet.
#
# Normally submitted automatically by submit_yt8m_pipeline.sh after the
# build job completes. Can also be run standalone if indexes are already built:
#   sbatch scripts/idun/yt8m_cpu_search.slurm
#
# NOTE: running serial + parallel roughly doubles wall time vs. the old
# single-mode run — increase --time below if jobs time out on 4m/8m sizes.
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
mkdir -p logs

module purge
module load GCCcore/11.3.0
module load CUDA/11.8.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

DATASETS="video1m video2m video4m video8m audi1m audi2m audi4m audi8m"

echo "--- serial run ---"
bash scripts/run_searches.sh --mode cpu_serial   --datasets $DATASETS

echo "--- parallel run ---"
bash scripts/run_searches.sh --mode cpu_parallel --datasets $DATASETS

echo "=========================================="
echo "CPU search complete: $(date)"
echo "=========================================="
