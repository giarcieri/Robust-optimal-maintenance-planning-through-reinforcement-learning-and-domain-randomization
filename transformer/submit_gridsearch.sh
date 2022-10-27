#!/bin/bash

#SBATCH -A es_chatzi
##SBATCH -G 1
#SBATCH -n 2
#SBATCH --time=130:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --job-name=submit-gridsearch
#SBATCH --output=transformer/gridsearch_results/output_gridsearch.txt

source /cluster/apps/local/env2lmod.sh
module load gcc/8.2.0 cuda/11.3.1 cudnn/8.2.1.32

python -m transformer.run_gridsearch \
--seed $1 \
--train_episodes $2 \
--test_episodes $3 \
--update_iterations $4 \
--gradient_descent_epochs $5 \
--num_heads $6 \
--num_layers $7 \
--hidden_sizes_mlp $8 \
--learning_rate $9 \
--alpha ${10} \
--save_rewards ${11} \
--save_model ${12} \
--gridsearch ${13} \
--keep_last_window_lenght_obs ${14} \
--polyak ${15} \
--replay_size ${16}
