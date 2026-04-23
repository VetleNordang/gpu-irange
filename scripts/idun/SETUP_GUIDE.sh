#!/bin/bash
#
# IDUN Job Submission Guide for iRange/RFANN Project
#
# This script demonstrates how to submit jobs to IDUN supercomputer
# and includes helpful commands for monitoring and managing jobs.
#

# ============================================================================
# STEP 1: Setup - Before First Submission
# ============================================================================

echo "=== STEP 1: IDUN SETUP ===" 

# Find your account(s) - run this command on IDUN:
echo "To find your account(s), run on IDUN login node:"
echo "  sacctmgr show assoc format=Account%15,User,QOS | grep -e QOS -e \$USER"
echo ""

# Check your CPU quota usage
echo "To check your CPU quota usage:"
echo "  idun-slurm-quota"
echo ""

# ============================================================================
# STEP 2: Modify Job Scripts
# ============================================================================

echo "=== STEP 2: MODIFY JOB SCRIPTS ===" 

echo "Edit the .slurm files in /scripts/idun/ and replace:"
echo "  1. <YOUR_ACCOUNT> - Replace with your actual account name"
echo "  2. <YOUR_EMAIL>   - Replace with your email (for job notifications)"
echo ""
echo "Example:"
echo "  sed -i 's/<YOUR_ACCOUNT>/my-dep/g' gpu_job.slurm"
echo "  sed -i 's/<YOUR_EMAIL>/name@ntnu.no/g' gpu_job.slurm"
echo ""

# ============================================================================
# STEP 3: Upload Project to IDUN
# ============================================================================

echo "=== STEP 3: UPLOAD PROJECT TO IDUN ===" 

echo "SCP your project to IDUN (from your local machine):"
echo "  scp -r /path/to/irange username@idun-login1.hpc.ntnu.no:~/"
echo ""
echo "Or use rsync for incremental uploads:"
echo "  rsync -avz --delete /path/to/irange/ username@idun-login1.hpc.ntnu.no:~/irange/"
echo ""

# ============================================================================
# STEP 4: Submit Jobs
# ============================================================================

echo "=== STEP 4: SUBMIT JOBS ===" 

echo "From IDUN login node, navigate to project:"
echo "  cd ~/irange"
echo ""

echo "Submit a GPU job (4 hours):"
echo "  sbatch scripts/idun/gpu_job.slurm"
echo ""

echo "Submit a CPU job (2 hours):"
echo "  sbatch scripts/idun/cpu_job.slurm"
echo ""

echo "Submit a quick test job (15 minutes, good for validation):"
echo "  sbatch scripts/idun/quick_test.slurm"
echo ""

# ============================================================================
# STEP 5: Monitor Jobs
# ============================================================================

echo "=== STEP 5: MONITOR JOBS ===" 

echo "View your jobs:"
echo "  squeue -u \$USER"
echo ""

echo "View only RUNNING jobs:"
echo "  squeue -u \$USER -t RUNNING"
echo ""

echo "View only PENDING jobs:"
echo "  squeue -u \$USER -t PENDING"
echo ""

echo "Get detailed info about a job:"
echo "  scontrol show jobid -dd <JOB_ID>"
echo ""

echo "View available resources:"
echo "  sinfo -o \"%10P %5D %34N  %5c  %7m  %47f  %23G\""
echo ""

# ============================================================================
# STEP 6: Cancel/Manage Jobs
# ============================================================================

echo "=== STEP 6: MANAGE JOBS ===" 

echo "Cancel a specific job:"
echo "  scancel <JOB_ID>"
echo ""

echo "Cancel all pending jobs:"
echo "  scancel -t PENDING -u \$USER"
echo ""

echo "Cancel all jobs:"
echo "  scancel -u \$USER"
echo ""

# ============================================================================
# STEP 7: Retrieve Results
# ============================================================================

echo "=== STEP 7: RETRIEVE RESULTS ===" 

echo "Download results back to local machine (from your local machine):"
echo "  scp -r username@idun-login1.hpc.ntnu.no:~/irange/logs ."
echo "  scp -r username@idun-login1.hpc.ntnu.no:~/irange/exectable_data/*/results ."
echo ""

# ============================================================================
# STEP 8: GPU and Resource Options
# ============================================================================

echo "=== STEP 8: GPU RESOURCE OPTIONS ===" 

echo "Available GPUs on IDUN:"
echo "  - p100  (16GB)"
echo "  - v100  (16GB or 32GB)"
echo "  - a100  (40GB or 80GB) - RECOMMENDED for modern workloads"
echo "  - h100  (80GB) - Most powerful, might have longer queue times"
echo ""

echo "Examples in job script:"
echo "  #SBATCH --gres=gpu:1           # Any available GPU"
echo "  #SBATCH --gres=gpu:a100:1      # 1 A100 GPU"
echo "  #SBATCH --gres=gpu:v100:2      # 2 V100 GPUs"
echo "  #SBATCH --constraint=gpu40g    # Only 40GB GPUs"
echo ""

# ============================================================================
# TIPS & BEST PRACTICES
# ============================================================================

echo "=== TIPS & BEST PRACTICES ===" 

echo "1. Time Limits:"
echo "   - Use 'short' partition for testing (20 min max)"
echo "   - Start with shorter times, increase if jobs time out"
echo "   - Can request time extension via help desk if needed"
echo ""

echo "2. CPU Cores:"
echo "   - CPUQ nodes typically have 28 cores"
echo "   - Set --cpus-per-task to match your program's parallelism"
echo "   - Disable hyperthreading if needed (typically not necessary)"
echo ""

echo "3. Memory:"
echo "   - Request what your job actually needs, not the maximum"
echo "   - Oversized memory requests increase queue wait times"
echo ""

echo "4. Build Strategy:"
echo "   - The scripts will rebuild if build/ doesn't exist"
echo "   - To force rebuild, remove build/ directory before submitting"
echo "   - Consider building on compute node vs. keeping pre-built binary"
echo ""

echo "5. Monitoring:"
echo "   - Check logs regularly: tail -f logs/gpu_*.log"
echo "   - Keep job output files for debugging"
echo ""

echo "6. Array Jobs (running many similar jobs):"
echo "   - Use #SBATCH --array=1-100 for 100 parallel tasks"
echo "   - Useful for batch experiments on different datasets"
echo ""

# ============================================================================
echo "=== READY TO SUBMIT! ===" 
echo "For more details, visit: https://www.hpc.ntnu.no/idun/documentation/running-jobs/"
echo "=========================================================================="
