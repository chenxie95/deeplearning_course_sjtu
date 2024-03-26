import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser()
parser.add_argument('--ar_cp', '-a', type=int, default=30)
parser.add_argument('--nar_cp', '-n', type=int, default=40)
parser.add_argument('--sample_p', '-p', type=float, default=0.99)
parser.add_argument('--run', '-r', type=int, default=0) #! set to -1 to unable multirun.
args = parser.parse_args()

# ! checkpoints selection
arcp = args.ar_cp
narcp = args.nar_cp

# ! inference settings
SAMPLE_P = args.sample_p
TEMPERATURE = 1

run = args.run

special_suffix = ""

model_name = f"valle-phone"
experiment_name = f"{model_name}-ar{arcp}-nar{narcp}-p{SAMPLE_P}-t{TEMPERATURE}"

experiment_name = experiment_name + bool(special_suffix)*"-" + special_suffix
generated_prefix = f"generated_all/{experiment_name}" if run == -1 else f"generated_runs/{experiment_name}/run{run}"

valid_sum, bad_sum = 0, 0
counter_paths = Path(generated_prefix, 'bad_counter').rglob('*.counter')
for counter_path in counter_paths:
    counter_dict = torch.load(counter_path)
    valid_sentences, bad_sentences = counter_dict['valid_sentences'], counter_dict['bad_sentences']
    valid_sum += valid_sentences
    bad_sum += bad_sentences

print("Source: ", generated_prefix)
print(f'total sentences: {valid_sum}, bad sentences: {bad_sum}, bad rate%: {bad_sum/valid_sum*100:.3f}')