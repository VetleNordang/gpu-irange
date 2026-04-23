# IDUN Supercomputer Job Submission Guide

This directory contains SLURM job scripts for running tests and experiments on the **IDUN supercomputer** at NTNU.

## Quick Start (3 Steps)

### 1. Find Your Account

On any IDUN login node, run:
```bash
sacctmgr show assoc format=Account%15,User,QOS | grep -e QOS -e $USER
```

You'll see output like:
```
        Account       User                  QOS 
         my-dep    USERNAME             normal 
   share-my-dep    USERNAME               high
```

### 2. Configure Job Scripts

From this directory (`scripts/idun/`), run:
```bash
./configure_jobs.sh my-dep name@ntnu.no
```

Or manually edit the `.slurm` files:
```bash
sed -i 's/<YOUR_ACCOUNT>/my-dep/g' *.slurm
sed -i 's/<YOUR_EMAIL>/name@ntnu.no/g' *.slurm
```

### 3. Submit a Job

```bash
# Quick test (15 min, good for validation)
sbatch quick_test.slurm

# GPU job (4 hours)
sbatch gpu_job.slurm

# CPU job (2 hours)
sbatch cpu_job.slurm
```

---

## Job Scripts Included

### `quick_test.slurm`
- **Partition**: short (testing)
- **Time**: 15 minutes
- **Resources**: 1 GPU, 4 CPU cores, 16GB RAM
- **Use for**: Quick validation, development, builds
- **Best for**: Fast iteration during development

### `gpu_job.slurm`
- **Partition**: GPUQ
- **Time**: 4 hours
- **Resources**: 1 GPU, 8 CPU cores, 32GB RAM
- **Use for**: CUDA/GPU experiments and full tests
- **Best for**: Main research computations

### `cpu_job.slurm`
- **Partition**: CPUQ
- **Time**: 2 hours
- **Resources**: 28 CPU cores, 64GB RAM (no GPU)
- **Use for**: CPU-only benchmarks, serial tests
- **Best for**: Non-GPU workloads

---

## Monitoring Jobs

### View your jobs
```bash
squeue -u $USER
```

### View only running jobs
```bash
squeue -u $USER -t RUNNING
```

### View only pending jobs
```bash
squeue -u $USER -t PENDING
```

### Check job details
```bash
scontrol show jobid -dd <JOB_ID>
```

### Check available resources
```bash
sinfo -o "%10P %5D %34N  %5c  %7m  %47f  %23G"
```

---

## Managing Jobs

### Cancel a job
```bash
scancel <JOB_ID>
```

### Cancel all pending jobs
```bash
scancel -t PENDING -u $USER
```

### Cancel all your jobs
```bash
scancel -u $USER
```

### Check CPU quota usage
```bash
idun-slurm-quota
```

---

## GPU Resource Options

Available GPUs on IDUN:

| GPU | Memory | Performance | Notes |
|-----|--------|-------------|-------|
| p100 | 16GB | Medium | Older, but reliable |
| v100 | 16GB/32GB | High | Good for most tasks |
| a100 | 40GB/80GB | Very High | **Recommended** for modern workloads |
| h100 | 80GB | Highest | Most powerful, longer queue |

### GPU Selection Examples

In your `.slurm` script, modify:
```bash
# Any available GPU
#SBATCH --gres=gpu:1

# Specific GPU type (1 A100)
#SBATCH --gres=gpu:a100:1

# Multiple GPUs (2 V100s)
#SBATCH --gres=gpu:v100:2

# With memory constraint (40GB or 80GB A100)
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint="gpu40g|gpu80g"
```

---

## File Transfer

### Upload to IDUN
```bash
# One-time copy
scp -r ./ username@idun-login1.hpc.ntnu.no:~/irange/

# Incremental sync (recommended)
rsync -avz --delete ./ username@idun-login1.hpc.ntnu.no:~/irange/
```

### Download results
```bash
scp -r username@idun-login1.hpc.ntnu.no:~/irange/logs ./
scp -r username@idun-login1.hpc.ntnu.no:~/irange/exectable_data/*/results ./
```

---

## Time Limits

- **short partition**: 20 minutes max (for testing)
- **CPUQ partition**: 30 days max
- **GPUQ partition**: 14 days max

If you need more time, contact the IDUN help desk with your job ID.

---

## Best Practices

1. **Start with short jobs** to validate your setup works
2. **Estimate time accurately** - oversized times increase wait times
3. **Request memory wisely** - only request what you need
4. **Use module system** - load modules in your job scripts:
   ```bash
   module purge
   module load CUDA/11.8.0
   ```
5. **Check logs regularly**:
   ```bash
   tail -f logs/gpu_*.log
   ```
6. **Test locally first** before submitting long jobs
7. **Keep output files** for debugging if jobs fail

---

## Troubleshooting

### Job stuck in PENDING
- Check available resources: `sinfo -o "%10P %5D %34N  %5c  %7m  %47f  %23G"`
- Reduce resource requests (cores, GPU type, memory)
- Try different GPU types

### Job times out
- Increase `--time` in your script
- Optimize your code for performance
- Request time extension from help desk

### Module not found
```bash
# Search for available modules
module spider python
module spider cuda

# Load specific version
module load CUDA/11.8.0
```

### Build fails on compute node
- Pre-build locally and copy binary
- Or ensure all dependencies are installed
- Check if needed compilers are loaded via modules

---

## Helper Scripts

### `configure_jobs.sh`
Automatically configures all job scripts with your account and email:
```bash
./configure_jobs.sh my-dep name@ntnu.no
```

### `SETUP_GUIDE.sh`
Prints comprehensive setup guide with examples.

---

## Useful Links

- **IDUN Documentation**: https://www.hpc.ntnu.no/idun/
- **Running Jobs Guide**: https://www.hpc.ntnu.no/idun/documentation/running-jobs/
- **Getting Started**: https://www.hpc.ntnu.no/idun/getting-started-on-idun/
- **SLURM Documentation**: https://slurm.schedmd.com/

---

## Array Jobs (Advanced)

For running many experiments, use job arrays:

```bash
#SBATCH --array=1-100
```

Then use `$SLURM_ARRAY_TASK_ID` in your script:
```bash
DATASET=$SLURM_ARRAY_TASK_ID
./build/tests/search --dataset=$DATASET
```

This submits 100 subjobs automatically.

---

## Questions?

1. Check IDUN documentation: https://www.hpc.ntnu.no/
2. Contact IDUN help desk (see website)
3. Check module system: `module avail`

---

**Last Updated**: 2026-04-22  
**Project**: iRange/RFANN  
**IDUN Partition Details**: https://www.hpc.ntnu.no/idun/
