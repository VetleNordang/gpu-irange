#!/bin/bash
#
# IDUN GPU PQ Search Job — GIST DATASETS ONLY.
#
# A focused, fast job to confirm the GPU PQ pipeline produces results on IDUN.
# Runs only the four gist datasets — all fit comfortably in ~6 GB VRAM, so any
# GPU in the constraint (including a 32 GB V100) works and scheduling stays fast.
# Once this completes cleanly, use gpu_pq_search_job.slurm for the full sweep.
#

#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g|gpu40g|gpu32g"
#SBATCH --job-name="irange_pq_gist"
#SBATCH --output=logs/gpu_pq_gist_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/idun/log_sysinfo.sh
echo "GPU PQ Search (gist only) started: $(date)"
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
# Build PQ binary
# ============================================================
echo "Building GPU PQ binary for $GPU_ARCH..."
rm -rf cude_version/build
if ! make -C cude_version pq_target \
        ARCH="-gencode arch=compute_${CC},code=sm_${CC}" \
        FAISS_INCLUDE="$FAISS_INCLUDE" \
        FAISS_LIB_PATH="$FAISS_LIB_PATH" \
        EXTRA_LDFLAGS="-Wl,-rpath,$CONDA_FAISS/lib -L$CONDA_FAISS/lib"; then
    echo "ERROR: pq_target build failed. Aborting."
    exit 1
fi

# ============================================================
# Run PQ searches — gist datasets only
# ============================================================
# --runs 1 : single pass (~10-15 min total) — a fast confirmation it works.
#            Change to 7 (1 warmup + 6 real) once you trust the pipeline.
# stdbuf   : force line-buffered stdout/stderr so the log shows live progress
#            instead of looking frozen at the memory-summary table.
echo ""
echo "=========================================="
echo "Running GPU PQ searches (gist only)..."
echo "=========================================="
stdbuf -oL -eL bash scripts/run_searches.sh --mode gpu_pq \
    --runs 1 \
    --datasets gist250k gist500k gist750k gist1000k

echo ""
echo "=========================================="
echo "GPU PQ searches (gist only) completed: $(date)"
echo "=========================================="
