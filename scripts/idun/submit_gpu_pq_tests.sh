#!/bin/bash
#
# Submit GPU PQ search pipeline.
# All data and indexes must already be on IDUN.
# Run from the repo root:
#   bash scripts/idun/submit_gpu_pq_tests.sh
#

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

GPU_PQ_JOB=$(sbatch --parsable scripts/idun/gpu_pq_search_job.slurm)
echo "GPU PQ search job submitted: $GPU_PQ_JOB"
echo "  log: logs/gpu_pq_search_${GPU_PQ_JOB}.log"
echo ""
echo "Check queue:  squeue -u \$USER"
echo "Cancel:       scancel $GPU_PQ_JOB"
