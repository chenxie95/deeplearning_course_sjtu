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
generated_prefix = f"./generated_non-continuation/{experiment_name}" if run == -1 else f"generated_non-continuation/{experiment_name}/run{run}"

p_sum, cut_p_sum = 0, 0
dict_paths = Path(generated_prefix, 'idx2gpath').rglob('*.ptdict')
idx2path_dict = {}
for dict_path in dict_paths:
    idx2path_dict_i = torch.load(dict_path)
    idx2path_dict.update(idx2path_dict_i)

torch.save(idx2path_dict, Path(generated_prefix, 'idx2gpath', 'all.ptdict'))