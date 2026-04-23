#!/bin/bash
#
# IDUN Short Test Job Script for iRange/RFANN project
# For quick development/testing (20 minute time limit)
# Useful for validating builds and running small tests
#

#SBATCH --partition=short             # Short partition (20 min limit, good for testing)
#SBATCH --account=<YOUR_ACCOUNT>      # TODO: Replace with your account
#SBATCH --time=00:15:00               # 15 minutes
#SBATCH --nodes=1                     # 1 compute node
#SBATCH --ntasks-per-node=1           # 1 task per node
#SBATCH --cpus-per-task=4             # 4 CPU cores
#SBATCH --mem=16G                     # 16GB memory
#SBATCH --gres=gpu:1                  # Optional: 1 GPU for quick GPU testing
#SBATCH --job-name="irange_test"
#SBATCH --output=logs/test_%j.log

# Print job information
echo "=========================================="
echo "Test job started on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Hostname: $(hostname)"
echo "=========================================="

# Set working directory
cd "$SLURM_SUBMIT_DIR"

# Load necessary modules
module purge
module load CUDA/11.8.0 GCCcore/11.3.0

# Quick rebuild
if [ -d "build" ]; then
    rm -rf build
fi
mkdir -p build
cd build

echo "Building project..."
cmake ..
make -j $SLURM_CPUS_PER_TASK

echo ""
echo "Running quick tests..."
echo "=========================================="

if [ -f "tests/search" ]; then
    echo "Running search test..."
    ./tests/search
fi

echo "=========================================="
echo "Test job completed on $(date)"
