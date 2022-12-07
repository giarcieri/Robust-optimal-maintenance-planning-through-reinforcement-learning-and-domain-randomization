#!/bin/bash

#SBATCH -A es_chatzi
##SBATCH -G 1
#SBATCH -n 2
#SBATCH --time=400:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=lstm-submit-gridsearch
#SBATCH --output=lstm/gridsearch_results/output_gridsearch.txt

source /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0 cuda/11.3.1 cudnn/8.2.1.32

python -m lstm.run_gridsearch \
--seed $1 \
--train_episodes $2 \
--test_episodes $3 \
--update_iterations $4 \
--gradient_descent_epochs $5 \
--hidden_sizes $6 \
--learning_rate $7 \
--alpha $8 \
--save_rewards $9 \
--save_model ${10} \
--gridsearch ${11} \
--polyak ${12} \
--replay_size ${13} 
