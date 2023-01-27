#bsub -o "transformer/gridsearch_results/output_gridsearch.txt" -n 1 -R "rusage[mem=2048]" python -m transformer.submit_gridsearch
#sbatch -o "transformer/gridsearch_results/output_gridsearch.txt" -n 1 --mem-per-cpu=2048 --wrap "python -m transformer.submit_gridsearch"
import subprocess

params = {
    "seed": [0, 732, 318, 698],
    "train_episodes": [10000, 20000, 40000, 60000, 80000], #10000
    "test_episodes": [500], #500
    "update_iterations": [10], #[1, 10]
    "gradient_descent_epochs": [1], #[1, 10]
    "num_heads": [8], #[2, 4, 8]
    "num_layers": [2], #[2, 4, 8]
    "hidden_sizes_mlp": [[100]], #[[], [100], [100, 100]]
    "learning_rate": [1e-3], #[1e-3, 5e-4, 1e-4]
    "alpha": [0.1, 1.],  #[0.2, 0.1]
    "save_rewards": [False],
    "save_model": [False],
    "gridsearch": [True],
    "keep_last_window_lenght_obs": [True], #[True, False]
    "polyak": [0.9], #[0.995, 0.9]
    "replay_size": [int(1e6)]
}

# Create all possible permutations
combinations = []
for v0 in params['seed']:
    for v1 in params['train_episodes']:
        for v2 in params['test_episodes']:
            for v3 in params['update_iterations']:
                for v4 in params['gradient_descent_epochs']:
                    for v5 in params['num_heads']:
                        for v6 in params['num_layers']:
                            for v7 in params['hidden_sizes_mlp']:
                                for v8 in params['learning_rate']:
                                    for v9 in params['alpha']:
                                        for v10 in params['save_rewards']:
                                            for v11 in params['save_model']:
                                                for v12 in params['gridsearch']:
                                                    for v13 in params['keep_last_window_lenght_obs']:
                                                        for v14 in params['polyak']:
                                                            for v15 in params['replay_size']:
                                                                combinations.append([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15])

for i, combo in enumerate(combinations):
    command = ['sbatch'] + ['transformer/submit_gridsearch.sh'] + [str(x) for x in combo] 
    #command = ['python'] + ['-m'] + ['transformer.run_gridsearch'] + inputs
    out = subprocess.run(command)