#!/bin/bash
#
# IDUN CPU Search Job: Run iRangeGraph search on all GIST indexes
#

#SBATCH --partition=CPUQ
#SBATCH --account=share-ie-idi
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64G
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

module purge
module load GCCcore/11.3.0
module load CUDA/11.8.0

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

if [ ! -f "build/tests/search" ]; then
    echo "ERROR: search binary not found. Run cpu_job.slurm first."
    exit 1
fi

bash run_all_searches.sh

echo "=========================================="
echo "CPU Search completed: $(date)"
echo "=========================================="
