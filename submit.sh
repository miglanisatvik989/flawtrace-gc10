#!/bin/bash
#PBS -N flawtrace
#PBS -q gpu
#PBS -j oe
#PBS -V
#PBS -l select=1:ncpus=16:ngpus=1:mem=32gb

cd "$PBS_O_WORKDIR"

# Initialize Conda
source /home/soft/anaconda3/etc/profile.d/conda.sh

# Activate environment
conda activate flawtrace-gc10

# Load CUDA
module load cuda/12.8

# Run training
python3 -u ./scripts/train.py --config configs/hpc.yaml > _main_train.log 2>&1

# Submit:
# qsub submit.sh

# Check status:
# qstat -f job_id
# qstat -u "$USER"

# Monitor training:
# tail -f _main_train.log