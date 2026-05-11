#!/bin/bash
#
# IDUN CPU Search Job: Run iRangeGraph search on all GIST indexes
#

#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --job-name="irange_cpu_search"
#SBATCH --output=logs/cpu_search_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

echo "=========================================="
echo "CPU Search started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node:   $(hostname)"
echo "Cores:  $SLURM_CPUS_PER_TASK"
echo "=========================================="

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load GCCcore/11.3.0
module load CUDA/12.1.1

CONDA_FAISS=/cluster/home/vetlean/.conda/envs/irange
export LD_LIBRARY_PATH="$CONDA_FAISS/lib:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ ! -f "build/tests/search" ]; then
    echo "search binary not found — building with cmake..."
    mkdir -p build
    cd build
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DFAISS_INCLUDE_DIR="$CONDA_FAISS/include" \
        -DFAISS_LIBRARY="$CONDA_FAISS/lib/libfaiss.so" \
        || { echo "ERROR: cmake failed"; exit 1; }
    make -j"$SLURM_CPUS_PER_TASK" search \
        || { echo "ERROR: make failed"; exit 1; }
    cd ..
    echo "Build complete."
fi

# NOTE: running serial then parallel roughly doubles wall time vs. the old
# single-mode run — increase --time above if jobs time out on large datasets.
echo "--- serial run ---"
bash scripts/run_searches.sh --mode cpu_serial   --datasets gist250k gist500k gist750k gist1000k

echo "--- parallel run ---"
bash scripts/run_searches.sh --mode cpu_parallel --datasets gist250k gist500k gist750k gist1000k

echo "=========================================="
echo "CPU Search completed: $(date)"
echo "=========================================="
