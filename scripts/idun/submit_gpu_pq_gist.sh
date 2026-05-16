#!/bin/bash
#
# Submit GPU PQ search pipeline — GIST DATASETS ONLY.
# A focused, fast job to confirm the GPU PQ pipeline produces results on IDUN.
# All gist data, indexes, and PQ files must already be on IDUN.
# Run from the repo root:
#   bash scripts/idun/submit_gpu_pq_gist.sh
#

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

GPU_PQ_GIST_JOB=$(sbatch --parsable scripts/idun/gpu_pq_gist_job.slurm)
echo "GPU PQ gist job submitted: $GPU_PQ_GIST_JOB"
echo "  log: logs/gpu_pq_gist_${GPU_PQ_GIST_JOB}.log"
echo ""
echo "Watch log:    tail -f logs/gpu_pq_gist_${GPU_PQ_GIST_JOB}.log"
echo "Watch CSVs:   watch ls executable_data/gist1m/250k/results/gpu_pq/"
echo "Check queue:  squeue -u \$USER"
echo "Cancel:       scancel $GPU_PQ_GIST_JOB"
