#!/bin/bash
#
#SBATCH --job-name=LOGSUMEXP-{{dataset}}
#SBATCH --output=logs/logs_lse_{{dataset}}_%A_%a.txt  # output file
#SBATCH -e logs/logs_lse_{{dataset}}_%A_%a.err        # File to which STDERR will be written
#SBATCH --mem=100000
#SBATCH --gres=gpu:1
#SBATCH --partition=m40-long

# To run use 
# $ cat run.sh  | sed -e 's/{{dataset}}/med/g' | sbatch 
# $ cat run.sh  | sed -e 's/{{dataset}}/WN18RR/g' | sbatch 
# $ cat run.sh  | sed -e 's/{{dataset}}/NELL-995/g' | sbatch 
python -m code.cbr --dataset={{dataset}} --max_num_programs=2 --n_paths=100 --train 1
