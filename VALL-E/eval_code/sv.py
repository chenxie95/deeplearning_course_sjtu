from pathlib import Path

import torch
import torchaudio
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector

from rich.progress import Progress
from rich.progress import SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

import argparse

PROP_SEC = 3
THRESHOLD = 0.86

parser = argparse.ArgumentParser()
parser.add_argument('--ar_cp', '-a', type=int, default=30)
parser.add_argument('--nar_cp', '-n', type=int, default=40)
parser.add_argument('--sample_p', '-p', type=float, default=1.0)
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

audio_prefix = "/home/ubuntu/LibriSpeech"
model_name = f"valle-phone"
experiment_name = f"{model_name}-ar{arcp}-nar{narcp}-p{SAMPLE_P}-t{TEMPERATURE}"

experiment_name = experiment_name + bool(special_suffix)*"-" + special_suffix
generated_prefix = f"generated_all/{experiment_name}" if run == -1 else f"generated_runs/{experiment_name}/run{run}"

generated_folder_names = [f'{generated_prefix}/test-clean']#, f'{generated_prefix}/test-other']

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
model = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv')

def compare_audio_one_pair(audio1, audio2):

    # audio files are decoded on the fly
    # @params [(m1, ), (m2, ) , ...]
    
    audio = [audio1.numpy(), audio2.numpy()]  
    inputs = feature_extractor(audio, padding=True, return_tensors="pt", sampling_rate=16000)
    with torch.no_grad():
        embeddings = model(**inputs).embeddings
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()

        # the resulting embeddings can be used for cosine similarity-based retrieval
        cosine_sim = torch.nn.CosineSimilarity(dim=-1)
        similarity = cosine_sim(embeddings[0], embeddings[1])
        if not (0 < similarity.item() < 1):
            return -1
    #threshold = 0.86  # the optimal threshold is dataset-dependent
    #if similarity < threshold:
    #    print("Speakers are not the same!")
    return similarity.item()


similaritys = []
gen_paths = []

for generated_folder_name in generated_folder_names:
    generated_paths = Path(generated_folder_name).rglob("*.gen")
    generated_paths = list(generated_paths)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        "Elapsed:", TimeElapsedColumn(),
        MofNCompleteColumn(),
    )
    task1 = progress.add_task("Audio loop", total=len(generated_paths))
    
    with progress:
        for idx, generated_path in enumerate(generated_paths):
            progress.update(task1, advance=1)

            rel_path = generated_path.relative_to(generated_prefix)

            orig_wav, orig_sr = torchaudio.load(str(Path(audio_prefix, rel_path).with_suffix('.flac'))) # ! check orig_sr = 16000
            assert orig_sr == 16000
            gen_wav, gen_sr = torchaudio.load(str(generated_path))

            sim = compare_audio_one_pair(gen_wav[0], orig_wav[0, :orig_sr*PROP_SEC])
            if sim == -1:
                print(f"Sim inf: {generated_path}; skipped.")
                continue
            similaritys.append(sim)
            gen_paths.append(rel_path)
            
    progress.stop()

assert len(similaritys) == len(gen_paths)

similaritys = torch.tensor(similaritys)
print("Source: ", generated_prefix)
print("# of sentences: ", similaritys.size(0))
print(f"Mean similarity: {similaritys.mean().item():.4f}")
print("Accept rate:", (similaritys>=THRESHOLD).sum()/similaritys.size(0))

results_prefix = Path(f'{generated_prefix}/results')
if not results_prefix.exists():
    results_prefix.mkdir(parents=True)

torch.save(similaritys, Path(results_prefix, 'sim_tensor.spk'))

with open(Path(results_prefix, 'path_sim.spk'), 'w', encoding='utf-8') as f:
    for i, gen_path in enumerate(gen_paths):
        print(f"{gen_path}\t{similaritys[i].item():.6f}", file=f)

sim_dict = {}
for i, gen_path in enumerate(gen_paths):
    sim_dict[gen_path.stem] = similaritys[i].item()

torch.save(sim_dict, Path(results_prefix, 'sim_dict.spk'))