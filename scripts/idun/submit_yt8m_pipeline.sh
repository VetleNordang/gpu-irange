#!/bin/bash
#
# Submit the full YouTube-8M pipeline: build indexes → CPU search → GPU search.
#
# Each step is held until the previous one succeeds.
# If any step fails, all later steps are cancelled automatically.
#
# Run from the repo root:
#   bash scripts/idun/submit_yt8m_pipeline.sh
#

set -e
cd "$(dirname "$0")/../.."   # repo root

mkdir -p logs

BUILD_JOB=$(sbatch --parsable scripts/idun/yt8m_build_indexes.slurm)
echo "Build job submitted:  $BUILD_JOB"
echo "  log: logs/yt8m_build_${BUILD_JOB}.log"

CPU_JOB=$(sbatch --parsable \
    --dependency=afterok:$BUILD_JOB \
    scripts/idun/yt8m_cpu_search.slurm)
echo "CPU search job submitted: $CPU_JOB  (starts after $BUILD_JOB)"
echo "  log: logs/yt8m_search_${CPU_JOB}.log"

GPU_JOB=$(sbatch --parsable \
    --dependency=afterok:$CPU_JOB \
    scripts/idun/gpu_search_job.slurm)
echo "GPU search job submitted: $GPU_JOB  (starts after $CPU_JOB)"
echo "  log: logs/gpu_search_${GPU_JOB}.log"

echo ""
echo "Check queue:  squeue -u \$USER"
echo "Cancel all:   scancel $BUILD_JOB $CPU_JOB $GPU_JOB"
