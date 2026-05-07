#!/bin/bash
#
# Submit GPU search pipeline.
# All data and indexes must already be on IDUN.
# Run from the repo root:
#   bash scripts/idun/submit_gpu_tests.sh
#

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

GPU_JOB=$(sbatch --parsable scripts/idun/gpu_search_job.slurm)
echo "GPU search job submitted: $GPU_JOB"
echo "  log: logs/gpu_search_${GPU_JOB}.log"
echo ""
echo "Check queue:  squeue -u \$USER"
echo "Cancel:       scancel $GPU_JOB"
