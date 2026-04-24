#!/bin/bash
#
# IDUN GPU Job Script for iRange/RFANN project
# For running CUDA/GPU tests and experiments
#

#SBATCH --partition=GPUQ              # GPU partition
#SBATCH --account=share-ie-idi
#SBATCH --time=04:00:00               # 4 hours (adjust as needed)
#SBATCH --nodes=1                     # 1 compute node
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --cpus-per-task=8             # 8 CPU cores
#SBATCH --mem=32G                     # 32GB memory
#SBATCH --gres=gpu:1                  # 1 GPU (use gpu:a100:1 or gpu:v100:1 for specific types)
#SBATCH --job-name="irange_gpu_test"
#SBATCH --output=logs/gpu_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=ALL

# Print job information
echo "=========================================="
echo "Job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "Working directory: $SLURM_SUBMIT_DIR"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "CPU cores per task: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Set working directory
cd "$SLURM_SUBMIT_DIR"

# Load necessary modules (adjust versions as needed)
module purge
module load CUDA/11.8.0

# Build the project if needed
if [ ! -d "build" ]; then
    echo "Building project..."
    mkdir -p build
    cd build
    cmake ..
    make -j $SLURM_CPUS_PER_TASK
    cd ..
else
    echo "Using existing build directory..."
fi

# Run GPU searches and save results
echo ""
echo "Running GPU searches..."
echo "=========================================="

bash run_all_searches_gpu.sh

echo "=========================================="
echo "Job completed on $(date)"
