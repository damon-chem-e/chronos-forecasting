#!/bin/bash
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -t 10
#SBATCH -p sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB

# Function to add directories to the PATH
pathadd() {
    if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
        PATH="${PATH:+"$PATH:"}$1"
    fi
}
pathadd "/home/adewinmb/.local/bin"

module load anaconda3/2023.07
source /home/adewinmb/.bashrc
source activate finetune

CUDA_VISIBLE_DEVICES=0 python \
    training/train.py --config training/configs/chronos-t5-small-distls.yaml \
    --model-id amazon/chronos-t5-small \
    --no-random-init \
    --max-steps 1000 \
    --learning-rate 0.001