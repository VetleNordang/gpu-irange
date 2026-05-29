#!/bin/bash
#
# Submit the full pipeline job to IDUN.
#
# Usage (from repo root):
#   bash scripts/idun/submit_full_pipeline.sh
#   bash scripts/idun/submit_full_pipeline.sh --runs 3
#   bash scripts/idun/submit_full_pipeline.sh --datasets audi1m video1m
#
# All extra arguments are forwarded to run_full_pipeline.sh inside the job.
#

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

JOB=$(sbatch --parsable scripts/idun/full_pipeline_job.slurm "$@")
echo "Full pipeline job submitted: $JOB"
echo "  log: logs/full_pipeline_${JOB}.log"
echo ""
echo "Check queue:  squeue -u \$USER"
echo "Cancel:       scancel $JOB"
