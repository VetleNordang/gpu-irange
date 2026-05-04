# IDUN Job Scripts

This folder contains SLURM job scripts and pipeline launchers for running iRangeGraph experiments on the IDUN supercomputer at NTNU.

---

## Files Overview

### Pipeline launchers (run these)

| Script | What it does |
|---|---|
| `submit_gist_pipeline.sh` | Submits the full GIST1M pipeline (4 chained jobs) |
| `submit_yt8m_pipeline.sh` | Submits the full YouTube-8M pipeline (4 chained jobs) |

### Individual SLURM jobs

| File | Partition | Time | What it does |
|---|---|---|---|
| `download_and_prepare_gist.slurm` | CPUQ | 2h | Downloads GIST1M from ftp.irisa.fr and converts `.fvecs` → `.bin` |
| `gist_build_indexes.slurm` | CPUQ | 4h | Builds HNSW indexes for all 4 GIST sizes (250k / 500k / 750k / 1000k) |
| `cpu_search_job.slurm` | CPUQ | 4h | Runs CPU search on all built GIST indexes |
| `gpu_search_job.slurm` | GPUQ | 3h | Runs GPU search on all GIST indexes (requires CUDA build) |
| `download_and_prepare_youtube8m.slurm` | CPUQ | 8h | Downloads YouTube-8M TFRecords and converts to `.bin` |
| `yt8m_build_indexes.slurm` | CPUQ | 4h | Builds HNSW indexes for YouTube-8M (video + audio) |
| `yt8m_cpu_search.slurm` | CPUQ | 4h | Runs CPU search on YouTube-8M indexes |

### Helper scripts

| Script | What it does |
|---|---|
| `configure_jobs.sh` | Sets account and email in all `.slurm` files at once |

---

## Pipelines

### GIST1M pipeline

```
download_and_prepare_gist  →  gist_build_indexes  →  cpu_search_job  →  gpu_search_job
       (CPUQ, 2h)                 (CPUQ, 4h)             (CPUQ, 4h)        (GPUQ, 3h)
```

Each job starts automatically only after the previous one succeeds. If any step fails, all later steps are cancelled.

**Submit from the repo root:**
```bash
bash scripts/idun/submit_gist_pipeline.sh
```

Output:
```
Download job submitted:     12345   log: logs/gist_prepare_12345.log
Build job submitted:        12346   (starts after 12345)
CPU search job submitted:   12347   (starts after 12346)
GPU search job submitted:   12348   (starts after 12347)

Check queue:  squeue -u $USER
Cancel all:   scancel 12345 12346 12347 12348
```

---

### YouTube-8M pipeline

```
download_and_prepare_youtube8m  →  yt8m_build_indexes  →  yt8m_cpu_search  →  gpu_search_job
          (CPUQ, 8h)                   (CPUQ, 4h)            (CPUQ, 4h)         (GPUQ, 3h)
```

**Submit from the repo root:**
```bash
bash scripts/idun/submit_yt8m_pipeline.sh
```

---

## First-time setup on IDUN

### 1. Upload the repo

From your local machine:
```bash
rsync -avz --delete ./ vetlean@idun-login1.hpc.ntnu.no:~/irange/
```

### 2. SSH in and build the binaries

```bash
ssh vetlean@idun-login1.hpc.ntnu.no
cd ~/irange
module purge
module load GCCcore/11.3.0 CUDA/12.1.1
make
```

### 3. Submit a pipeline

```bash
bash scripts/idun/submit_gist_pipeline.sh
```

---

## Monitoring jobs

```bash
squeue -u $USER                          # all your jobs
squeue -u $USER -t RUNNING               # running only
tail -f logs/gist_prepare_<JOB_ID>.log  # live log output
scancel <JOB_ID>                         # cancel one job
scancel -u $USER                         # cancel all your jobs
```

---

## Results location

After the pipelines finish, results are written to:

```
exectable_data/gist1m/250k/results/
exectable_data/gist1m/500k/results/
exectable_data/gist1m/750k/results/
exectable_data/gist1m/1000k/results/
exectable_data/video/1m/results/
exectable_data/audi/1m/results/
```

Logs are in `logs/` named by job type and SLURM job ID.

---

## GPU notes

The GPU job (`gpu_search_job.slurm`) requests `--exclusive` so it gets the full GPU node without interference. It checks for at least 10 GB of free GPU memory before running and auto-detects the GPU compute capability (`sm_70`, `sm_80`, etc.) at runtime.

If the GPU binary is not already built, the job builds it automatically before running.
