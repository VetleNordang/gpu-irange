#!/bin/bash
#
# IDUN Index Build Job: Build HNSW indexes for all GIST1M sizes
#

#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --job-name="gist_build_indexes"
#SBATCH --output=logs/gist_build_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "GIST Index Build started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Cores:  $SLURM_CPUS_PER_TASK"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load GCCcore/11.3.0
module load CUDA/12.1.1

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ ! -f "build/tests/buildindex" ]; then
    echo "buildindex binary not found — building..."
    make -C . buildindex || { echo "ERROR: build failed"; exit 1; }
fi

bash build_gist_indexes.sh

echo "=========================================="
echo "GIST Index Build completed: $(date)"
echo "=========================================="
