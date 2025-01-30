#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 60
#SBATCH -p sched_mit_sloan_gpu
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

python evaluation/evaluate.py evaluation/configs/zero-shot.yaml evaluation/results/chronos-t5-small-zero-shot.csv \
    --chronos-model-id "output/run-62658600-23-0/checkpoint-final" \
    --eval-dataset-name "monash_tourism_yearly" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20
    