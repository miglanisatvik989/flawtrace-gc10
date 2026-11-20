#!/bin/bash
#PBS -N flawtrace-eval
#PBS -q gpu
#PBS -j oe
#PBS -V
#PBS -l select=1:ncpus=16:ngpus=1:mem=32gb

set -e
cd "${PBS_O_WORKDIR:?}"
exec > _main_eval.log 2>&1

source /home/soft/anaconda3/etc/profile.d/conda.sh
conda activate flawtrace-gc10
module load cuda/12.8

device=$(python3 -c 'import yaml; print(yaml.safe_load(open("configs/hpc.yaml"))["DEVICE"])')
exec python3 -u scripts/evaluate.py \
    --checkpoint experiments/hpc-full-06/weights/best.pt \
    --device "$device" --split val --run-id "${EVAL_RUN_ID:-b1-val-03}"
