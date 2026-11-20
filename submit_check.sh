#!/bin/bash -l
#PBS -N flawtrace-check
#PBS -q gpu
#PBS -l select=1:ncpus=2:ngpus=1:mem=8gb
#PBS -l walltime=00:10:00
#PBS -j oe

set -eo pipefail
cd "${PBS_O_WORKDIR:?Submit from the FlawTrace repository root}"
mkdir -p logs
{
    source /home/soft/anaconda3/etc/profile.d/conda.sh
    conda activate flawtrace-gc10
    module load cuda/12.8
    printf 'Host %s | CUDA_VISIBLE_DEVICES=%s\n' "$(hostname)" "${CUDA_VISIBLE_DEVICES-<unset>}"
    nvidia-smi --query-gpu=index,uuid,name,driver_version,memory.total --format=csv
    nvcc --version
    df -h /dev/shm
    python -c 'import torch; n = torch.cuda.device_count(); print(f"Visible CUDA devices: {n}"); raise SystemExit(0 if n == 1 else "Expected exactly one PBS-assigned CUDA device.")'
    python -u scripts/check_env.py
} 2>&1 | tee "logs/${PBS_JOBID:?}.log"
