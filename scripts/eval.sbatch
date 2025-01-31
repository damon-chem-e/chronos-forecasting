#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 60
#SBATCH -p sched_mit_sloan_gpu_r8
#SBATCH --gres=gpu:1
#SBATCH --mem=8GB

# Function to add directories to the PATH
pathadd() {
    if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
        PATH="${PATH:+"$PATH:"}$1"
    fi
}
pathadd "/home/adewinmb/.local/bin"

module load anaconda3/2023.07
source /home/adewinmb/.bashrc
source activate chronos2

python evaluation/evaluate.py evaluation/configs/zero-shot.yaml evaluation/results/dominick_test.csv \
    --chronos-model-id "output/run-62739985-0-0/checkpoint-final" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20
    