#!/bin/bash
#
# Quick script to configure IDUN job scripts with your account info
#
# Usage: ./configure_jobs.sh <account> <email>
# Example: ./configure_jobs.sh my-dep name@ntnu.no
#

if [ $# -ne 2 ]; then
    echo "Usage: $0 <account> <email>"
    echo ""
    echo "Example:"
    echo "  $0 my-dep name@ntnu.no"
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
echo "Email: $EMAIL"
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
echo "1. Upload your project to IDUN: rsync -avz --delete ./ username@idun-login1.hpc.ntnu.no:~/irange/"
echo "2. SSH into IDUN: ssh username@idun-login1.hpc.ntnu.no"
echo "3. Submit a job: cd ~/irange && sbatch scripts/idun/quick_test.slurm"
echo "4. Monitor: squeue -u \$USER"
