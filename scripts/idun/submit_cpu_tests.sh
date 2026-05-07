#!/bin/bash
#
# Submit CPU search pipeline.
# All data and indexes must already be on IDUN.
# Run from the repo root:
#   bash scripts/idun/submit_cpu_tests.sh
#

set -e
cd "$(dirname "$0")/../.."

mkdir -p logs

CPU_JOB=$(sbatch --parsable scripts/idun/cpu_search_job.slurm)
echo "CPU search job submitted: $CPU_JOB"
echo "  log: logs/cpu_search_${CPU_JOB}.log"
echo ""
echo "Check queue:  squeue -u \$USER"
echo "Cancel:       scancel $CPU_JOB"
