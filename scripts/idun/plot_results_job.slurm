#!/bin/bash
#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name="irange_plot"
#SBATCH --output=logs/plot_%j.log
#SBATCH --mail-user=nordang.vetle@gmail.com
#SBATCH --mail-type=END,FAIL

echo "Plot job started: $(date)"
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

module purge
module load Anaconda3/2024.02-1

CONDA_PYTHON=/cluster/home/vetlean/.conda/envs/irange/bin/python3

"$CONDA_PYTHON" python/plots/plot_gpu_vs_cpu.py --env idun

echo "Plot job done: $(date)"
