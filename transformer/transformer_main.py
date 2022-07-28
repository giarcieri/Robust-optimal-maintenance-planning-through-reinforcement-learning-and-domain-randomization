from .transformer_run_loop import *
import json
import argparse
from functools import partial

parser = argparse.ArgumentParser(description='VIBD')
parser.add_argument('-n', '--n_seeds', type=int, metavar='',
                    required=True, help='seed')
args = parser.parse_args()

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

with open('transformer/config.json') as config_file:
    config = json.load(config_file)

devices = jax.local_device_count()
with open("transformer/logs.txt", "a") as f:
    f.write(f"Running on {devices} parallel devices\n")
seeds = int(args.n_seeds)
with open("transformer/logs.txt", "a") as f:
    f.write(f"Running {seeds} parallel seeds\n")
seeds = jnp.broadcast_to(jnp.arange(seeds), shape=(seeds,2)).astype(jnp.uint32)
run_loop_partial = partial(run_loop, trace=trace, reward_matrix=reward_matrix, **config)
run_loop_pmap = jax.vmap(run_loop_partial, axis_name='i')
run_loop_pmap(seeds)