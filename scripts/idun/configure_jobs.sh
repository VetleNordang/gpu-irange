#!/bin/bash
#
# Quick script to configure IDUN job scripts with your account info
#
# Usage: ./configure_jobs.sh <account> <email>
# Example: ./configure_jobs.sh share-ie-idi name@ntnu.no
#
# To find your account, run on IDUN:
#   sacctmgr show assoc format=Account%15,User,QOS | grep -e QOS -e $USER
#

if [ $# -ne 2 ]; then
    echo "Usage: $0 <account> <email>"
    echo ""
    echo "Example:"
    echo "  $0 share-ie-idi name@ntnu.no"
    echo ""
    echo "To find your account, run on IDUN:"
    echo "  sacctmgr show assoc format=Account%15,User,QOS | grep -e QOS -e \$USER"
    exit 1
fi

ACCOUNT="$1"
EMAIL="$2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Configuring IDUN job scripts..."
echo "Account: $ACCOUNT"
echo "Email:   $EMAIL"
echo ""

# Configure all .slurm files
for slurm_file in "$SCRIPT_DIR"/*.slurm; do
    if [ -f "$slurm_file" ]; then
        filename=$(basename "$slurm_file")
        echo "Configuring $filename..."
        sed -i "s/<YOUR_ACCOUNT>/$ACCOUNT/g" "$slurm_file"
        sed -i "s/<YOUR_EMAIL>/$EMAIL/g" "$slurm_file"
    fi
done

echo ""
echo "✓ Configuration complete!"
echo ""
echo "Next steps:"
echo "1. Upload your project to IDUN: rsync -avz --delete ./ vetlean@idun-login1.hpc.ntnu.no:~/gpu-irange/"
echo "2. SSH into IDUN: ssh vetlean@idun-login1.hpc.ntnu.no"
echo "3. Submit the GIST pipeline: cd ~/gpu-irange && bash scripts/idun/submit_gist_pipeline.sh"
echo "4. Submit the YT8M pipeline: cd ~/gpu-irange && bash scripts/idun/submit_yt8m_pipeline.sh"
echo "5. Monitor: squeue -u \$USER"
