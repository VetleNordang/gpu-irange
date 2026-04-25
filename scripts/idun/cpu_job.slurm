#!/bin/bash
#
# IDUN CPU Job Script for iRange/RFANN project
# For running CPU-only tests and experiments
#

#SBATCH --partition=CPUQ              # CPU partition
#SBATCH --account=share-ie-idi
#SBATCH --time=02:00:00               # 2 hours
#SBATCH --nodes=1                     # 1 compute node
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --cpus-per-task=28            # 28 CPU cores (typical CPUQ node)
#SBATCH --mem=64G                     # 64GB memory
#SBATCH --job-name="irange_cpu_test"
#SBATCH --output=logs/cpu_%j.log
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
echo "CPU cores: $SLURM_CPUS_PER_TASK"
echo "=========================================="

# Set working directory
cd "$SLURM_SUBMIT_DIR"

# Load necessary modules
module purge
module load GCCcore/12.2.0
module load CMake/3.24.3-GCCcore-12.2.0

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

# Build GIST indexes
echo ""
echo "Building GIST indexes..."
echo "=========================================="

bash build_gist_indexes.sh

echo "=========================================="
echo "Job completed on $(date)"
