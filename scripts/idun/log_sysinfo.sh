#!/bin/bash
# Source this file to get log_cpu_sysinfo and log_gpu_sysinfo functions.
# Usage in SLURM scripts:
#   source "$(dirname "${BASH_SOURCE[0]}")/log_sysinfo.sh"
#   log_cpu_sysinfo   # CPU jobs
#   log_gpu_sysinfo   # GPU jobs (prints CPU info + GPU info)

log_cpu_sysinfo() {
    local os_name kernel cpu_model cpu_sockets cpu_cores_per_socket cpu_threads mem_total

    os_name=$(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2 | tr -d '"' || echo "unknown")
    kernel=$(uname -r)
    cpu_model=$(lscpu 2>/dev/null | grep "^Model name" | sed 's/Model name:[[:space:]]*//' | xargs)
    cpu_sockets=$(lscpu 2>/dev/null | awk '/^Socket\(s\)/{print $2}')
    cpu_cores_per_socket=$(lscpu 2>/dev/null | awk '/^Core\(s\) per socket/{print $NF}')
    cpu_threads=$(lscpu 2>/dev/null | awk '/^Thread\(s\) per core/{print $NF}')
    local physical=$(( ${cpu_sockets:-1} * ${cpu_cores_per_socket:-1} ))
    local logical=$(( physical * ${cpu_threads:-1} ))
    mem_total=$(awk '/MemTotal/{printf "%.0f GB", $2/1024/1024}' /proc/meminfo 2>/dev/null || echo "unknown")

    echo "=========================================="
    echo "System Info"
    echo "=========================================="
    echo "Date:          $(date)"
    echo "Hostname:      $(hostname)"
    echo "OS:            $os_name"
    echo "Kernel:        $kernel"
    echo "CPU:           $cpu_model"
    echo "CPU cores:     $physical physical, $logical logical (${cpu_sockets:-?} socket(s))"
    echo "SLURM cores:   ${SLURM_CPUS_PER_TASK:-N/A}"
    echo "Memory:        $mem_total"
    echo "SLURM job:     ${SLURM_JOB_ID:-N/A}"
    echo "SLURM node:    ${SLURM_NODELIST:-$(hostname)}"
    echo "Git commit:    $(git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    echo "=========================================="
}

log_gpu_sysinfo() {
    log_cpu_sysinfo

    echo "GPU Info"
    echo "=========================================="
    if command -v nvidia-smi &>/dev/null; then
        local gpu_name gpu_driver gpu_vram gpu_cap
        gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
        gpu_driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | xargs)
        gpu_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | xargs)
        gpu_cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | xargs)
        local sm_arch="sm_$(echo "$gpu_cap" | tr -d '.')"

        echo "GPU:           $gpu_name"
        echo "VRAM:          $gpu_vram"
        echo "Driver:        $gpu_driver"
        echo "Compute cap:   $gpu_cap  ($sm_arch)"

        local cuda_ver
        cuda_ver=$(nvcc --version 2>/dev/null | grep "release" | sed 's/.*release //;s/,.*//' | xargs)
        [[ -z "$cuda_ver" ]] && cuda_ver=$(nvidia-smi 2>/dev/null | awk '/CUDA Version/{print $NF}')
        echo "CUDA:          ${cuda_ver:-unknown}"

        local gpu_count
        gpu_count=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "GPU count:     $gpu_count"
    else
        echo "nvidia-smi not available"
    fi
    echo "=========================================="
}
