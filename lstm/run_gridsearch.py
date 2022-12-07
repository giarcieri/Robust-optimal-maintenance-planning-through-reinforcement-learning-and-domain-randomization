import argparse
import pickle
import ast
from .lstm_run_loop import *
from jax.lib import xla_bridge

parser = argparse.ArgumentParser(description='rlfr')
parser.add_argument('-i', '--seed', type=int, metavar='',
                    required=True)
parser.add_argument('-tr', '--train_episodes', type=int, metavar='',
                    required=True)
parser.add_argument('-te', '--test_episodes', type=int, metavar='',
                    required=True)
parser.add_argument('-ui', '--update_iterations', type=int, metavar='',
                    required=True)
parser.add_argument('-gd', '--gradient_descent_epochs', type=int, metavar='',
                    required=True)
parser.add_argument('-hs', '--hidden_sizes', type=str, metavar='',
                    required=True)
parser.add_argument('-lr', '--learning_rate', type=float, metavar='',
                    required=True)
parser.add_argument('-a', '--alpha', type=float, metavar='',
                    required=True)
parser.add_argument('-sr', '--save_rewards')
parser.add_argument('-sm', '--save_model')
parser.add_argument('-gr', '--gridsearch')
parser.add_argument('-pk', '--polyak', type=float, metavar='',
                    required=True)
parser.add_argument('-rs', '--replay_size', type=int, metavar='',
                    required=True)
args = parser.parse_args()


args.hidden_sizes = ast.literal_eval(args.hidden_sizes)
# convert to dictionary
config = vars(args)

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

print("Running on", xla_bridge.get_backend().platform)

run_loop(trace=trace, reward_matrix=reward_matrix, domain_randomization=False, window_length=5, 
        domain_randomization_test=False, **config)