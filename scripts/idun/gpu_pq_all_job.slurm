#!/bin/bash
#
# IDUN GPU PQ Search Job — ALL DATASETS.
#
# Runs the GPU PQ search across every dataset (gist x4, video x4, audi x4).
# PQ only — no gpu_normal pass. A clean from-scratch rebuild of hello_pq
# guarantees the binary matches the current source, so a stale binary can
# never be reused.
#
# Note: the 8M video/audi datasets are large. Their visited arrays and
# adjacency lists may not fit on a 32 GB GPU. If those datasets OOM, drop
# gpu32g from the constraint so the job lands on a 40 GB or 80 GB GPU.
#

#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g|gpu32g"
#SBATCH --job-name="irange_pq_all"
#SBATCH --output=logs/gpu_pq_all_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/idun/log_sysinfo.sh
echo "GPU PQ Search (all datasets) started: $(date)"
log_gpu_sysinfo

module purge
module load GCCcore/11.3.0
module load CUDA/12.1.1
module load Anaconda3/2024.02-1

CONDA_FAISS=/cluster/home/vetlean/.conda/envs/irange
export FAISS_INCLUDE="-I$CONDA_FAISS/include"
export FAISS_LIB_PATH="-L$CONDA_FAISS/lib"
export LD_LIBRARY_PATH="$CONDA_FAISS/lib:$LD_LIBRARY_PATH"
export LIBRARY_PATH="$CONDA_FAISS/lib:$LIBRARY_PATH"
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
# Build PQ binary — clean from-scratch rebuild
# ============================================================
echo "Building GPU PQ binary for $GPU_ARCH..."
rm -rf cude_version/build
if ! make -C cude_version pq_target \
        ARCH="-gencode arch=compute_${CC},code=sm_${CC}" \
        FAISS_INCLUDE="$FAISS_INCLUDE" \
        FAISS_LIB_PATH="$FAISS_LIB_PATH"; then
    echo "ERROR: pq_target build failed. Aborting."
    exit 1
fi

# ============================================================
# Run PQ searches — all datasets
# ============================================================
# No --datasets flag → all 12 (gist x4, video x4, audi x4).
# --runs 7 : 1 warmup + 6 measured runs → averaged results for algorithm testing.
# Datasets missing their index or PQ files are skipped, not failed.
echo ""
echo "=========================================="
echo "Running GPU PQ searches (all datasets)..."
echo "=========================================="
bash scripts/run_searches.sh --mode gpu_pq --runs 7

echo ""
echo "=========================================="
echo "GPU PQ searches (all datasets) completed: $(date)"
echo "=========================================="
