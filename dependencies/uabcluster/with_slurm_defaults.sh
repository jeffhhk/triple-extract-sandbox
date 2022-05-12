#!/bin/bash
#SLURM_NTASKS aka --ntasks
#SLURM_CPUS_PER_TASK aka --cpus-per-task
#SLURM_MEM_PER_CPU aka --mem-per-cpu
#SBATCH_TIMELIMIT aka --time
#SBATCH_PARTITION aka --partition
#SBATCH_JOB_NAME aka --job-name
#SBATCH_GRES aka --gres

env \
SLURM_NTASKS=1 \
SLURM_CPUS_PER_TASK=1 \
SLURM_MEM_PER_CPU=4096 \
SBATCH_TIMELIMIT=08:00:00 \
SBATCH_PARTITION=pascalnodes \
SBATCH_JOB_NAME=hello_gpu \
SBATCH_GRES=gpu:1 \
"$@"