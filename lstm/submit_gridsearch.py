#bsub -o "lstm/gridsearch_results/output_gridsearch.txt" -n 1 -R "rusage[mem=2048]" python -m lstm.submit_gridsearch
#sbatch -o "lstm/gridsearch_results/output_gridsearch.txt" -n 1 --mem-per-cpu=2048 --wrap "python -m lstm.submit_gridsearch"
import subprocess

params = {
    "seed": [0, 732, 100, 29],
    "train_episodes": [60000], 
    "test_episodes": [500], 
    "update_iterations": [10],
    "gradient_descent_epochs": [10],
    "hidden_sizes": [[100, 100, 100]],
    "learning_rate": [1e-3],
    "alpha": [0.1], 
    "save_rewards": [False],
    "save_model": [False],
    "gridsearch": [True],
    "polyak": [0.995],
    "replay_size": [int(1e6)],
    "window_length": [2, 3, 4, 5, 10],
}

# Create all possible permutations
combinations = []
for v0 in params['seed']:
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
                                                    for v13 in params['window_length']:
                                                        combinations.append([v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13])

for i, combo in enumerate(combinations):
    inputs = []
    for v, x in zip(params.keys(), combo):
        inputs.append(f'--{v}')
        inputs.append(str(x))
    command = ['bsub'] + ['-o'] + ['lstm/gridsearch_results/output_gridsearch.txt'] + ['-n'] + ['2'] + ['-W'] + \
     ['300:00'] + ['-R'] + ['rusage[mem=8192]'] + ['python'] + ['-m'] + ['lstm.run_gridsearch'] + inputs
    #command = ['sbatch'] + ['lstm/submit_gridsearch.sh'] + [str(x) for x in combo] 
    #command = ['bash'] + ['lstm/submit_gridsearch.sh'] + [str(x) for x in combo] 
    out = subprocess.run(command)