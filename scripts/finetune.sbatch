#!/bin/bash
#SBATCH -N 2
#SBATCH -n 4
#SBATCH -t 30
#SBATCH -p sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --mem=16GB

module load anaconda3/2023.07

# Function to add directories to the PATH
pathadd() {
    if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
        PATH="${PATH:+"$PATH:"}$1"
    fi
}
pathadd "/home/adewinmb/.local/bin"

source /home/adewinmb/.bashrc
source activate finetune

CUDA_VISIBLE_DEVICES=0 python \
    training/train.py --config training/configs/chronos-t5-small-distls.yaml \
    --model-id amazon/chronos-t5-small \
    --no-random-init \
    --max-steps 1000 \
    --learning-rate 0.001