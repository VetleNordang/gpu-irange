# The Simple Guide to NVIDIA GPU Profiling

When writing CUDA code, you need to know if your code is running efficiently. NVIDIA provides two distinct tools for this, and you should use them in a specific order:

1. **Nsight Systems (`nsys`)** - The **Macro** view (The "When"). 
2. **Nsight Compute (`ncu`)** - The **Micro** view (The "Why").

Here is the simple, no-script approach to using them.

---

## Tool 1: Nsight Systems (`nsys`)
**What it does:** It records a timeline of your entire program. It shows you what the CPU is doing, when data is copied to the GPU, and when the GPU is doing work. 

**What you are looking for:** "Bubbles" (empty spaces) where the GPU is doing nothing because it's waiting for the CPU or waiting for data.

### 1. How to run it
Run this directly in your terminal (no script needed). Just put `nsys profile -o <filename>` in front of your normal run command:

```bash
cd /workspaces/irange/cude_version
nsys profile -o my_timeline_profile ./build/hello --data_path ../exectable_data/gist1m/500k/gist_base_500k.bin --query_path ../exectable_data/gist1m/500k/gist_query_500k.bin --M 32
```

### 2. Where is it saved?
It creates a file called `my_timeline_profile.nsys-rep` in your current folder. 

### 3. How to evaluate it
1. Copy the file to your local computer: `scp user@server:/path/to/my_timeline_profile.nsys-rep ~/Downloads/`
2. Open it locally using the GUI: `nsys-ui ~/Downloads/my_timeline_profile.nsys-rep`
3. **What to look for in the GUI:**
   * **GPU Idle Time:** Look at the CUDA hardware row. Are there massive gaps between green blocks? If yes, your CPU is too slow at feeding the GPU.
   * **Memory Transfers:** Are the red/brown blocks (Memcpy) taking longer than the green blocks (Kernels)? If yes, you are bottlenecked by PCI-e bandwidth, not by GPU math.
   * **Synchronizations:** Is almost all your time spent in `cudaDeviceSynchronize`? If yes, you are forcing the CPU to wait for the GPU too often instead of doing parallel work.

---

## Tool 2: Legacy Profiler (`nvprof`)

**What it does:** It zooms in on a *single* GPU function (kernel). It tells you exactly how efficient the math is, how well you are using the GPU's memory cache, and if your threads are structured correctly.

**IMPORTANT HARDWARE NOTE:** Your machine has a **Tesla P100** (Pascal Architecture). Nsight Compute (`ncu`) does **NOT** support Pascal architectures anymore. If you try to use `ncu`, it just runs the application and says `==WARNING== No kernels were profiled.`. We *must* use the legacy tool `nvprof` instead!

### 1. How to run it
Put `nvprof --metrics all` in front of your executable. We will also add `--csv --log-file my_kernel_metrics.csv` so it saves the data to a permanent spreadsheet you can read later!

**Important Time-Saver:** Because measuring hardware metrics slows down the GPU drastically, we don't want to measure all 544 test runs. Nvprof's built-in kernel filter can be buggy, so the absolute best way to only profile the first 11 runs is to **temporarily edit your C++ testing loop**!

1. Open `hello.cu` (or whichever file contains your test loop).
2. Put a variable like `int run_count = 0;` before the loop.
3. Inside the loop, put `run_count++; if (run_count >= 11) break;`.
4. Recompile with `make all`.

Then, you can safely run the normal profiler command without any messy regex filters, and it will only take a couple of seconds to get the 11 profiles!

```bash
cd /workspaces/irange/cude_version

# First, make sure it's compiled:
make all

# Then profile the executable (which now stops itself after 11 loops):
nvprof --metrics all --csv --log-file my_kernel_metrics.csv ./build/hello \
    --data_path ../exectable_data/gist1m/500k/gist_base_500k.bin \
    --query_path ../exectable_data/gist1m/500k/gist_query_500k.bin \
    --range_saveprefix ../exectable_data/gist1m/500k/query_ranges/query_ranges_500k \
    --groundtruth_saveprefix ../exectable_data/gist1m/500k/groundtruth/groundtruth_500k \
    --index_file ../exectable_data/gist1m/500k/gist_500k.index \
    --result_saveprefix ../exectable_data/gist1m/500k/results/results_500k \
    --M 32
```

*(Note: Gathering `--metrics all` takes extra time because nvprof runs your kernel multiple times to read all the different hardware sensors. If it takes too long, you can use narrower metrics like `nvprof --metrics achieved_occupancy,sm_efficiency`)*

### 2. Reading the output
Because we added `--csv --log-file my_kernel_metrics.csv`, it will save all the hardware metrics into a file named `my_kernel_metrics.csv` inside your `cude_version` directory. You can download this file and open it in Excel, Google Sheets, or just view it in VS Code!

### 3. How to evaluate it
Open the generated CSV-file. Here is what to look for to find weaknesses:

* **Memory Workload Analysis:** 
  You will see metrics regarding `dram_read_throughput` and `dram_write_throughput`. 
  * If the bandwidth matches the limit of your card, your kernel is **Memory Bound**.
* **Occupancy:**
  It will show `achieved_occupancy`. If this is low (e.g., 20%), you are not giving the GPU enough work to do at once, or your kernel is using too many "registers" (local variables) per thread, which prevents the GPU from scheduling more blocks.

---

## The Ultimate Workflow
1. Use **`nsys`** to find out *which* kernel is taking up the most time, and to make sure your CPU isn't the real problem.
2. Once you find the slow kernel and confirm data is getting to the GPU properly, use **`nvprof`** on a small dataset to understand *why* that specific kernel is slow.
3. Fix the code.
4. Repeat.