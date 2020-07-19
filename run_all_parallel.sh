#!/bin/bash
#
#SBATCH --job-name=CBR_MED
#SBATCH --output=logs/logs_med_%A_%a.txt  # output file
#SBATCH -e logs/logs_med_%A_%a.err        # File to which STDERR will be written
#SBATCH --mem=50000
#SBATCH --array=0-19
#SBATCH --cpus-per-task=12
#SBATCH --ntasks=1
#SBATCH --partition=longq


splitsize=$(expr 3430 / 20 + 1)
python -m code.cbr --dataset_name=med --k_adj=5 --max_num_programs=25 --test --parallelize --splitid=$SLURM_ARRAY_TASK_ID --splitsize=$splitsize
