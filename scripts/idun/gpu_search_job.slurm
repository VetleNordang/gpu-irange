#!/bin/bash
#
# IDUN GPU Search Job: Run iRangeGraph GPU search on all GIST indexes
# Targets A100, V100, or H100 GPUs (not P100)
#

#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100|v100|h100"
#SBATCH --job-name="irange_gpu_search"
#SBATCH --output=logs/gpu_search_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "GPU Search started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"

module purge
module load GCCcore/11.3.0
module load CUDA/11.8.0

CONDA_FAISS=/cluster/home/vetlean/.conda/envs/irange
export FAISS_INCLUDE="-I$CONDA_FAISS/include"
export FAISS_LIB_PATH="-L$CONDA_FAISS/lib"
export LD_LIBRARY_PATH="$CONDA_FAISS/lib:$LD_LIBRARY_PATH"

# Use the conda env's Python so pandas/matplotlib are available for plots
export PATH="$CONDA_FAISS/bin:$PATH"

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Detect GPU compute capability (e.g. 7.0 -> sm_70)
CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
export GPU_ARCH="sm_${CC}"
echo "Compute capability: $GPU_ARCH"

if [ ! -f "build/tests/search" ]; then
    echo "ERROR: search binary not found. Run cpu_job.slurm first."
    exit 1
fi

echo ""
echo "Running GPU searches..."
echo "=========================================="

bash run_all_searches_gpu.sh

echo "=========================================="
echo "GPU Search completed: $(date)"
echo "=========================================="
