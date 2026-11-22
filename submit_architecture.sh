#!/bin/bash -l
#PBS -N flawtrace-arch
#PBS -q gpu
#PBS -l select=1:ncpus=16:ngpus=1:mem=32gb
#PBS -j oe

cd "$PBS_O_WORKDIR"

source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
module load cuda/12.8

python -u scripts/check_architecture.py --device cuda:0
