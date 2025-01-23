#!/bin/bash 
#SBATCH -n 1 #Request 1 task (cores)
#SBATCH -N 1 #Request 1 node
#SBATCH -t 0-5:00 #Request runtime of 5 hours
#SBATCH -p sched_mit_sloan_gpu #Run on sched_mit_sloan_gpu partition
#SBATCH --mem-per-gpu=80GB #Request 80GB

#SBATCH -o output_%j.txt #redirect output to output_JOBID.txt
#SBATCH -e error_%j.txt #redirect errors to error_JOBID.txt
#SBATCH --mail-type=BEGIN,END #Mail when job starts and ends
#SBATCH --mail-user=test@test.com #email recipient
echo "Hello World"
