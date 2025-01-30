#!/bin/bash
#SBATCH -p sched_mit_sloan_gpu
#SBATCH --job-name=finetune_all         # Job name
#SBATCH --output=output_%A_%a.out       # Standard output log (%A = Job ID, %a = Array index)
#SBATCH --error=error_%A_%a.err         # Standard error log
#SBATCH --array=0-26                    # Array range (adjust based on the number of arguments)
#SBATCH --time=01:00:00                 # Max run time (hh:mm:ss)
#SBATCH --mem=8G                        # Memory per job
#SBATCH --cpus-per-task=4               # Number of CPUs per job
#SBATCH --gres=gpu:1

# Function to add directories to the PATH
pathadd() {
    if [ -d "$1" ] && [[ ":$PATH:" != *":$1:"* ]]; then
        PATH="${PATH:+"$PATH:"}$1"
    fi
}
pathadd "/home/adewinmb/.local/bin"

module load anaconda3/2023.07
source /home/adewinmb/.bashrc
source activate chronos

LINE_NUM=$((SLURM_ARRAY_TASK_ID * 2 + 1))  # Adjust index to read pairs
DATASET=$(sed -n "${LINE_NUM}p" datasets.txt)
PRED_LEN=$(sed -n "$((LINE_NUM + 1))p" datasets.txt)

CUDA_VISIBLE_DEVICES=0 python \
    training/train.py --config training/configs/chronos-t5-small-distls.yaml \
    --training-data-paths "['preprocessing_training_data/zero_shot_datasets_train_val/train/$DATASET']" \
    --prediction-length $PRED_LEN \
    --base-fname "run-$SLURM_JOB_ID-$SLURM_ARRAY_TASK_ID" \
    --model-id amazon/chronos-t5-small \
    --no-random-init \
    --max-steps 1000 \
    --learning-rate 0.001
