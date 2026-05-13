#!/bin/bash
#
# IDUN PQ Training Job: Train PQ models and encode all datasets (CPU).
# Skips datasets whose PQ files already exist.
#

#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name="irange_pq_train"
#SBATCH --output=logs/pq_training_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "PQ Training started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Cores:  $SLURM_CPUS_PER_TASK"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load GCCcore/11.3.0
module load CUDA/12.1.1
module load CMake/3.23.1-GCCcore-11.3.0

CONDA_FAISS=/cluster/home/vetlean/.conda/envs/irange
export LD_LIBRARY_PATH="$CONDA_FAISS/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ ! -f "build/tests/pq_compress_only" ]; then
    echo "pq_compress_only not found — building..."
    mkdir -p build && cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_FAISS=ON \
        -DFAISS_INCLUDE_DIR="$CONDA_FAISS/include" \
        -DFAISS_LIBRARY="$CONDA_FAISS/lib/libfaiss.so" \
        || { echo "ERROR: cmake failed"; exit 1; }
    make -j"$SLURM_CPUS_PER_TASK" pq_compress_only \
        || { echo "ERROR: make failed"; exit 1; }
    cd ..
    echo "Build complete."
fi

bash scripts/make_pq.sh all

echo "=========================================="
echo "PQ Training completed: $(date)"
echo "=========================================="
