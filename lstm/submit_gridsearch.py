#bsub -o "lstm/gridsearch_results/output_gridsearch.txt" -n 1 -R "rusage[mem=2048]" python -m lstm.submit_gridsearch
import subprocess

params = {
    "train_episodes": [10000], 
    "test_episodes": [500], 
    "update_iterations": [10, 20],
    "gradient_descent_epochs": [10, 20],
    "hidden_sizes": [[100, 100], [200, 200, 200]],
    "learning_rate": [1e-3],
    "alpha": [0.2, 0.01], 
    "save_rewards": [False],
    "save_model": [False],
    "gridsearch": [True],
    "polyak": [0.995, 0.7],
    "replay_size": [int(1e6), int(1e3)]
}

# Create all possible permutations
combinations = []
for v1 in params['train_episodes']:
    for v2 in params['test_episodes']:
        for v3 in params['update_iterations']:
            for v4 in params['gradient_descent_epochs']:
                for v5 in params['hidden_sizes']:
                    for v6 in params['learning_rate']:
                        for v7 in params['alpha']:
                            for v8 in params['save_rewards']:
                                for v9 in params['save_model']:
                                    for v10 in params['gridsearch']:
                                        for v11 in params['polyak']:
                                            for v12 in params['replay_size']:
                                                combinations.append([v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12])

for i, combo in enumerate(combinations):
    inputs = []
    for v, x in zip(params.keys(), combo):
        inputs.append(f'--{v}')
        inputs.append(str(x))
    command = ['bsub'] + ['-o'] + ['lstm/gridsearch_results/output_gridsearch.txt'] + ['-n'] + ['2'] + ['-W'] + \
     ['250:00'] + ['-R'] + ['rusage[mem=8192]'] + ['python'] + ['-m'] + ['lstm.run_gridsearch'] + \
        ['--seed'] + [str(0)] + inputs
    #command = ['python'] + ['-m'] + ['lstm.run_gridsearch'] + ['--seed'] + [str(0)] + inputs
    out = subprocess.run(command)