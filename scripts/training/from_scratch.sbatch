#!/bin/bash
#SBATCH --job-name=chronos-t5-tiny
#SBATCH -n 4 #Request 4 cores
#SBATCH --gres=gpu:1
#SBATCH -N 1 #Request 1 node
#SBATCH -p sched_mit_sloan_gpu #Run on sched_mit_sloan_gpu partition

#SBATCH -o output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e error_%j.txt #redirect errors to error_JOBID.txt

module load anaconda3/2023.07
source /home/hanngao/.bashrc
source activate nlp2
CUDA_VISIBLE_DEVICES=0 python train.py --config ./configs/chronos-t5-tiny.yaml