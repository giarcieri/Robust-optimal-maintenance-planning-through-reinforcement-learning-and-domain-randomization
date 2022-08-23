import json
import argparse
from functools import partial
import pickle
import jax

parser = argparse.ArgumentParser(description='rlfr')
parser.add_argument('-i', '--seed', type=int, metavar='',
                    required=True, help='seed')
parser.add_argument('-f', '--function', type=str, metavar='',
                    required=True, help='transformer or lstm')
args = parser.parse_args()

devices = jax.local_device_count()
seed = int(args.seed)

if args.function == 'transformer':
    from transformer.transformer_run_loop import *
    with open('transformer/config.json') as config_file:
        config = json.load(config_file)
    with open("transformer/logs.txt", "a") as f:
        f.write(f"Running seed {seed}\n")
elif args.function == 'lstm':
    from lstm.lstm_run_loop import *
    with open('lstm/config.json') as config_file:
        config = json.load(config_file)
    with open("lstm/logs.txt", "a") as f:
        f.write(f"Running seed {seed}\n")

file = 'trace.pickle'
with open(file, "rb") as fp:
    trace = pickle.load(fp)
    
reward_a_0 = - 0
reward_a_R2 = - 50
reward_a_A1 = - 2000 


reward_s_0 = - 100
reward_s_1 = - 200
reward_s_2 = - 1000
reward_s_3 = - 8000 

reward_matrix = jnp.asarray([
    [reward_a_0 + reward_s_0, reward_a_0 + reward_s_1, reward_a_0 + reward_s_2, reward_a_0 + reward_s_3],
    [reward_a_R2 + reward_s_0, reward_a_R2 + reward_s_1, reward_a_R2 + reward_s_2, reward_a_R2 + reward_s_3],
    [1*reward_a_A1 + reward_a_R2 + reward_s_0, 1.33*reward_a_A1 + reward_a_R2 + reward_s_1, 1.66*reward_a_A1 + reward_a_R2 + reward_s_2, 2*reward_a_A1 + reward_a_R2 + reward_s_3]
])

run_loop_partial = partial(run_loop, trace=trace, reward_matrix=reward_matrix, **config)
run_loop_partial(seed)