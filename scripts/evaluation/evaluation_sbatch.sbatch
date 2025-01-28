#!/bin/bash
#SBATCH --job-name=chronos-t5-tiny
#SBATCH -n 4 #Request 4 cores
#SBATCH --gres=gpu:1
#SBATCH -N 1 #Request 1 node
#SBATCH -p sched_mit_sloan_gpu_r8 #Run on sched_mit_sloan_gpu partition
#SBATCH --mem=250G

#SBATCH -o output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e error_%j.txt #redirect errors to error_JOBID.txt

module load anaconda3/2023.07
source /home/hanngao/.bashrc
source activate nlp2
python evaluate.py configs/zero-shot.yaml /home/hanngao/chronos-forecasting/scripts/evaluation/results /nfs/sloanlab007/projects/chimera_proj/models/run-ce-dominick/checkpoint-final