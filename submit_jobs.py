import subprocess
import argparse
import jax

parser = argparse.ArgumentParser(description='rlfr')
parser.add_argument('-n', '--n_seeds', type=int, metavar='',
                    required=True, help='number seeds')
parser.add_argument('-f', '--function', type=str, metavar='',
                    required=True, help='transformer or lstm')
args = parser.parse_args()

devices = jax.local_device_count()
seeds = int(args.n_seeds)

for i in range(seeds):
    command = ['bsub'] + ['-o'] + ['output.txt'] + ['-n'] + ['1'] + ['-W'] + ['200:00'] + \
        ['-R'] + ['rusage[mem=2048]'] + ['python'] + ['main.py'] +  ['--seed'] + [f'{i}'] + ['--function'] + [f'{args.function}']
    out = subprocess.run(command)
