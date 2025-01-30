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
source activate chronos2

# Read dataset name from dataset_names.txt based on array task ID
DATASET_NAME=$(sed -n "${SLURM_ARRAY_TASK_ID}p" dataset_names.txt)

python evaluation/evaluate.py evaluation/configs/zero-shot.yaml evaluation/results/$DATASET_NAME-$SLURM_ARRAY_TASK_ID.csv \
    --chronos-model-id "output/run-$SLURM_ARRAY_TASK_ID/checkpoint-final" \
    --batch-size=32 \
    --device=cuda:0 \
    --num-samples 20
