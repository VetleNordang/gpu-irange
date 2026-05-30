#!/bin/bash
#
# IDUN Dataset Check Job: Run integrity checks on all datasets.
#

#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --job-name="irange_dataset_check"
#SBATCH --output=logs/dataset_check_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

echo "Dataset check started: $(date)"
echo "Node: $(hostname)"

module purge
module load Anaconda3/2024.02-1

CONDA_PYTHON=/cluster/home/vetlean/.conda/envs/irange/bin/python3

echo "Running checks (skip-unique for speed)..."
"$CONDA_PYTHON" python/check_all_datasets.py

echo "=========================================="
echo "Dataset check completed: $(date)"
echo "=========================================="
