#!/bin/bash
#
# Download only results/ folders from IDUN into temp_idun_results/.
#
# Local layout mirrors IDUN:
#   temp_idun_results/audi/1m/results/
#   temp_idun_results/video/1m/results/
#   temp_idun_results/gist1m/1000k/results/
#   etc.
#
# Usage:
#   bash scripts/download_idun_results.sh [user@idun-login5.hpc.ntnu.no]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

IDUN_HOST="${1:-idun-login1.hpc.ntnu.no}"
IDUN_DATA="/cluster/home/vetlean/gpu-irange/executable_data"
LOCAL_DIR="$PROJECT_ROOT/temp_idun_results"

DATASETS=(
    "audi/1m"
    "audi/2m"
    "audi/4m"
    "audi/6m"
    "video/1m"
    "video/2m"
    "video/4m"
    "video/6m"
    "gist1m/250k"
    "gist1m/500k"
    "gist1m/750k"
    "gist1m/1000k"
)

echo "Downloading results from $IDUN_HOST"
echo "Remote: $IDUN_DATA"
echo "Local:  $LOCAL_DIR"
echo ""

any_failed=0

for ds in "${DATASETS[@]}"; do
    remote_results="$IDUN_DATA/$ds/results/"
    local_dest="$LOCAL_DIR/$ds/"

    # Check remote dir exists before trying
    if ! ssh "$IDUN_HOST" "test -d '$remote_results'" 2>/dev/null; then
        echo "SKIP: $ds/results/ not found on IDUN"
        continue
    fi

    echo "Syncing $ds/results/ ..."
    mkdir -p "$local_dest"

    if rsync -av --progress \
            "$IDUN_HOST:$remote_results" \
            "$local_dest/results/"; then
        echo "  OK: $ds"
    else
        echo "  FAILED: $ds"
        any_failed=1
    fi
    echo ""
done

echo "Done. Results in: $LOCAL_DIR"
[[ $any_failed -eq 0 ]] && exit 0 || exit 1
