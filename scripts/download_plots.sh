#!/bin/bash
# Download all analysis plots from IDUN to local executable_data/.
# Run from the project root on this server:
#   bash scripts/download_plots.sh

IDUN_USER="vetlean"
IDUN_HOST="idun-login1.hpc.ntnu.no"
IDUN_ROOT="~/gpu-irange/executable_data"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/executable_data"

echo "Downloading plots from $IDUN_HOST..."

rsync -avz --include="*/" --include="analysis/*.png" --exclude="*" \
    "$IDUN_USER@$IDUN_HOST:$IDUN_ROOT/" \
    "$LOCAL_ROOT/"

echo "Done. Plots saved under executable_data/*/results/analysis/"
