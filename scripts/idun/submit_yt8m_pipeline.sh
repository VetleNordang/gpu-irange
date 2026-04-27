#!/bin/bash
#
# Submit the full YouTube-8M pipeline: build indexes, then CPU search.
#
# The search job is automatically held until the build job succeeds.
# If the build job fails, the search job is cancelled automatically.
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

SEARCH_JOB=$(sbatch --parsable \
    --dependency=afterok:$BUILD_JOB \
    scripts/idun/yt8m_cpu_search.slurm)
echo "Search job submitted: $SEARCH_JOB  (starts after $BUILD_JOB finishes)"
echo "  log: logs/yt8m_search_${SEARCH_JOB}.log"

echo ""
echo "Check queue:  squeue -u \$USER"
echo "Cancel both:  scancel $BUILD_JOB $SEARCH_JOB"
