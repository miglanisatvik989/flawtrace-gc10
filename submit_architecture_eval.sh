#!/bin/bash
#PBS -N flawtrace-a1-eval
#PBS -q gpu
#PBS -j oe
#PBS -V
#PBS -l select=1:ncpus=16:ngpus=1:mem=32gb

set -e
cd "${PBS_O_WORKDIR:?}"
source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
module load cuda/12.8
exec python3 -u scripts/evaluate_architecture.py "$@"
