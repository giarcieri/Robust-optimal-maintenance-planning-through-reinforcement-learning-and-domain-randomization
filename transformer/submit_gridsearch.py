# Submit with bsub -o "gridsearch_results/output_gridsearch.txt" -n 1 -R "rusage[mem=2048]" python -u submit_gridsearch.py
import subprocess

params = {
    "train_episodes": [10000],
    "test_episodes": [500],
    #"keep_last_window_lenght_obs": [True, False],
    "update_iterations": [1, 10],
    "gradient_descent_epochs": [1, 10],
    "num_heads": [4, 8],
    "num_layers": [2, 8],
    "hidden_sizes_mlp": [[], [100]],
    "learning_rate": [1e-3, 1e-4],
    #"polyak": [0.995, 0.9],
    "alpha": [0.2, 0.1], 
    "save_rewards": [False],
    "save_model": [False],
    "gridsearch": [True]
}

# Create all possible permutations
combinations = []
for v1 in params['train_episodes']:
    for v2 in params['test_episodes']:
        for v3 in params['save_rewards']:
            for v4 in params['update_iterations']:
                for v5 in params['gradient_descent_epochs']:
                    for v6 in params['num_heads']:
                        for v7 in params['num_layers']:
                            for v8 in params['hidden_sizes_mlp']:
                                for v9 in params['learning_rate']:
                                    for v10 in params['save_model']:
                                        for v11 in params['alpha']:
                                            for v12 in params['gridsearch']:
                                                combinations.append([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12])

for i, combo in enumerate(combinations):
    command = ['bsub'] + ['-o'] + ['gridsearch_results/output_gridsearch.txt'] + ['-n'] + ['1'] + ['-W'] + ['200:00'] + \
        ['-R'] + ['rusage[mem=2048]'] + ['python'] + ['run_gridsearch.py'] + ['0'] + [str(x) for x in combo]
    out = subprocess.run(command)