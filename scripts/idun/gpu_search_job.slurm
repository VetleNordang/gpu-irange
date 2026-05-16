#!/bin/bash
#
# IDUN GPU Search Job: Run iRangeGraph GPU normal + PQ search on all datasets.
# Both binaries are built and run sequentially on the same 80 GB GPU node.
#

#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g|gpu32g"
#SBATCH --job-name="irange_gpu_all"
#SBATCH --output=logs/gpu_search_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/idun/log_sysinfo.sh
echo "GPU Search started: $(date)"
log_gpu_sysinfo

module purge
module load GCCcore/11.3.0
module load CUDA/12.1.1
module load Anaconda3/2024.02-1

CONDA_FAISS=/cluster/home/vetlean/.conda/envs/irange
export FAISS_INCLUDE="-I$CONDA_FAISS/include"
export FAISS_LIB_PATH="-L$CONDA_FAISS/lib"
export LD_LIBRARY_PATH="$CONDA_FAISS/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_FAISS/lib:$LIBRARY_PATH"   # compile-time — fixes GLIBCXX_3.4.30
export PATH="$CONDA_FAISS/bin:$PATH"
export CONDA_PYTHON="$CONDA_FAISS/bin/python3"

FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1 | tr -d ' MiB')
if [ "${FREE_MIB:-0}" -lt 10000 ]; then
    echo "ERROR: GPU has only ${FREE_MIB} MiB free — need at least 10 GB. Aborting."
    nvidia-smi
    exit 1
fi

CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' .')
export GPU_ARCH="sm_${CC}"
echo "Compute capability: $GPU_ARCH"

# ============================================================
# Build both binaries
# ============================================================
echo "Building GPU normal binary for $GPU_ARCH..."
rm -rf cude_version/build
if ! make -C cude_version optimized_test \
        ARCH="-gencode arch=compute_${CC},code=sm_${CC}" \
        FAISS_INCLUDE="$FAISS_INCLUDE" \
        FAISS_LIB_PATH="$FAISS_LIB_PATH"; then
    echo "ERROR: optimized_test build failed. Aborting."
    exit 1
fi

echo "Building GPU PQ binary for $GPU_ARCH..."
if ! make -C cude_version pq_target \
        ARCH="-gencode arch=compute_${CC},code=sm_${CC}" \
        FAISS_INCLUDE="$FAISS_INCLUDE" \
        FAISS_LIB_PATH="$FAISS_LIB_PATH" \
        EXTRA_LDFLAGS="-Wl,-rpath,$CONDA_FAISS/lib -L$CONDA_FAISS/lib"; then
    echo "ERROR: pq_target build failed. Aborting."
    exit 1
fi

# ============================================================
# Run searches
# ============================================================
echo ""
echo "=========================================="
echo "Running GPU normal searches..."
echo "=========================================="
bash scripts/run_searches.sh --mode gpu_normal

echo ""
echo "=========================================="
echo "Running GPU PQ searches..."
echo "=========================================="
bash scripts/run_searches.sh --mode gpu_pq

echo ""
echo "=========================================="
echo "All GPU searches completed: $(date)"
echo "=========================================="
