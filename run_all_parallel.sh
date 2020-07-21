#!/bin/bash
#
#SBATCH --job-name=CBR_MED
#SBATCH --output=logs/logs_med_parallel_%A_%a.txt  # output file
#SBATCH -e logs/logs_med_parallel_%A_%a.err        # File to which STDERR will be written
#SBATCH --mem=50000
#SBATCH --array=0-19
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --partition=longq


python -m code.cbr --dataset_name=NELL-995 --k_adj=5 --max_num_programs=15 --test --parallelize --splitid=$SLURM_ARRAY_TASK_ID --num_splits=20
