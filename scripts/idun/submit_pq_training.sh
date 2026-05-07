#!/bin/bash
#
# Submit PQ training pipeline.
# Trains PQ models for all datasets. Skips any that are already trained.
# Run from the repo root:
#   bash scripts/idun/submit_pq_training.sh
#

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

TRAIN_JOB=$(sbatch --parsable scripts/idun/pq_training_job.slurm)
echo "PQ training job submitted: $TRAIN_JOB"
echo "  log: logs/pq_training_${TRAIN_JOB}.log"
echo ""
echo "Check queue:  squeue -u \$USER"
echo "Cancel:       scancel $TRAIN_JOB"
echo ""
echo "After this completes, submit the PQ search:"
echo "  bash scripts/idun/submit_gpu_pq_tests.sh"
