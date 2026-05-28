#!/bin/bash
# Download all analysis plots from IDUN to local executable_data/.
# Run from the project root on this server:
#   bash scripts/download_plots.sh

IDUN_USER="vetlean"
IDUN_HOST="idun-login1.hpc.ntnu.no"
IDUN_ROOT="/cluster/home/vetlean/gpu-irange/executable_data"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/executable_data"

echo "Downloading plots from $IDUN_HOST..."

sync_analysis() {
    local remote_analysis="$1"
    local local_idun="$2"
    ssh "$IDUN_USER@$IDUN_HOST" "test -d $remote_analysis" 2>/dev/null || return
    mkdir -p "$local_idun"
    rsync -avz "$IDUN_USER@$IDUN_HOST:$remote_analysis/" "$local_idun/"
}

DATASETS=$(ssh "$IDUN_USER@$IDUN_HOST" "ls $IDUN_ROOT/")

for dataset in $DATASETS; do
    # dataset-level results
    sync_analysis \
        "$IDUN_ROOT/$dataset/results/analysis" \
        "$LOCAL_ROOT/$dataset/results/analysis/idun"

    # size-level results (e.g. 1m, 2m, 4m, 8m)
    SIZES=$(ssh "$IDUN_USER@$IDUN_HOST" "ls $IDUN_ROOT/$dataset/ 2>/dev/null" | grep -E '^[0-9]')
    for size in $SIZES; do
        sync_analysis \
            "$IDUN_ROOT/$dataset/$size/results/analysis" \
            "$LOCAL_ROOT/$dataset/$size/results/analysis/idun"
    done
done

echo "Done. Plots saved under executable_data/*/results/analysis/idun/ and executable_data/*/*/results/analysis/idun/"
