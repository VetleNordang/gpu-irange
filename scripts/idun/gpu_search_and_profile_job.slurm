#!/bin/bash
#
# IDUN GPU Search + Profiling Job
# Builds all GPU binaries, runs all GPU searches, then profiles
# gpu_normal and gpu_pq on audi1m and video1m at suffix 2 and 8.
#

#SBATCH --partition=GPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80g"
#SBATCH --job-name="irange_gpu_profile"
#SBATCH --output=logs/gpu_search_profile_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

source scripts/idun/log_sysinfo.sh
echo "Job started: $(date)"
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
echo "Compute capability: sm_${CC}"

# ============================================================
# Build all binaries
# ============================================================
echo ""
echo "=== Building binaries ==="
rm -rf cude_version/build

make -C cude_version optimized_test \
    ARCH="-gencode arch=compute_${CC},code=sm_${CC}" \
    FAISS_INCLUDE="$FAISS_INCLUDE" FAISS_LIB_PATH="$FAISS_LIB_PATH" \
    || { echo "ERROR: optimized_test build failed"; exit 1; }

make -C cude_version pq_target \
    ARCH="-gencode arch=compute_${CC},code=sm_${CC}" \
    FAISS_INCLUDE="$FAISS_INCLUDE" FAISS_LIB_PATH="$FAISS_LIB_PATH" \
    EXTRA_LDFLAGS="-Wl,-rpath,$CONDA_FAISS/lib -L$CONDA_FAISS/lib" \
    || { echo "ERROR: pq_target build failed"; exit 1; }

make -C cude_version root_target \
    ARCH="-gencode arch=compute_${CC},code=sm_${CC}" \
    FAISS_INCLUDE="$FAISS_INCLUDE" FAISS_LIB_PATH="$FAISS_LIB_PATH" \
    || { echo "ERROR: root_target build failed"; exit 1; }

# ============================================================
# Run all GPU searches
# ============================================================
echo ""
echo "=== Running GPU normal searches ==="
bash scripts/run_searches.sh --mode gpu_normal

echo ""
echo "=== Running GPU PQ searches ==="
bash scripts/run_searches.sh --mode gpu_pq

echo ""
echo "=== Running GPU root searches ==="
bash scripts/run_searches.sh --mode gpu_root

# ============================================================
# Profiling — gpu_normal and gpu_pq on audi1m and video1m
# ============================================================
echo ""
echo "=== Profiling ==="

PROFILE_DATASETS=(audi1m video1m)
PROFILE_SUFFIXES=(2 5 8)

for DATASET in "${PROFILE_DATASETS[@]}"; do
    for MODE in gpu_normal gpu_pq; do
        echo ""
        echo "--- Profiling $MODE on $DATASET ---"
        bash scripts/run_profiling.sh \
            --mode "$MODE" \
            --dataset "$DATASET" \
            --suffixes "${PROFILE_SUFFIXES[*]}"
    done
done

echo ""
echo "=== Job finished: $(date) ==="
