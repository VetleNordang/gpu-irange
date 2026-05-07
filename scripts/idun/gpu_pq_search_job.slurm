#!/bin/bash
#
# IDUN GPU PQ Search Job: Run iRangeGraph GPU PQ search on all indexes
#

#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --exclusive
#SBATCH --job-name="irange_gpu_pq_search"
#SBATCH --output=logs/gpu_pq_search_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "GPU PQ Search started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load GCCcore/11.3.0
module load CUDA/12.1.1
module load Anaconda3/2024.02-1

CONDA_FAISS=/cluster/home/vetlean/.conda/envs/irange
export FAISS_INCLUDE="-I$CONDA_FAISS/include"
export FAISS_LIB_PATH="-L$CONDA_FAISS/lib"
export LD_LIBRARY_PATH="$CONDA_FAISS/lib:$LD_LIBRARY_PATH"
export PATH="$CONDA_FAISS/bin:$PATH"

echo "GPU info:"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1 | tr -d ' MiB')
if [ "${FREE_MIB:-0}" -lt 10000 ]; then
    echo "ERROR: GPU has only ${FREE_MIB} MiB free — need at least 10 GB. Aborting."
    nvidia-smi
    exit 1
fi

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
export GPU_ARCH="sm_${CC}"
echo "Compute capability: $GPU_ARCH"

echo "Rebuilding GPU PQ binary for $GPU_ARCH..."
rm -f cude_version/build/hello_pq
if ! make -C cude_version pq_target \
        ARCH="-gencode arch=compute_${CC},code=sm_${CC}" \
        FAISS_INCLUDE="$FAISS_INCLUDE" \
        FAISS_LIB_PATH="$FAISS_LIB_PATH"; then
    echo "ERROR: build failed. Aborting."
    exit 1
fi

echo ""
echo "Running GPU PQ searches..."
echo "=========================================="

bash scripts/run_all_searches_gpu_pq.sh

echo "=========================================="
echo "GPU PQ Search completed: $(date)"
echo "=========================================="
