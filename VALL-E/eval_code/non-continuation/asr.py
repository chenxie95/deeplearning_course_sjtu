from pathlib import Path
from transformers import WhisperProcessor
from jiwer import compute_measures

import torch
import nemo.collections.asr as nemo_asr

from rich.progress import Progress
from rich.progress import SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

import argparse

cuda = 0

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

reference_prefix = "/home/ubuntu/LibriSpeech"
model_name = f"valle-phone"
experiment_name = f"{model_name}-ar{arcp}-nar{narcp}-p{SAMPLE_P}-t{TEMPERATURE}"

experiment_name = experiment_name + bool(special_suffix)*"-" + special_suffix
generated_prefix = f"./generated_non-continuation/{experiment_name}" if run == -1 else f"generated_non-continuation/{experiment_name}/run{run}"

generated_folder_names = [f'{generated_prefix}/test-clean']#, f'{generated_prefix}/test-other']

processor = WhisperProcessor.from_pretrained("openai/whisper-base")
model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained("nvidia/stt_en_conformer_transducer_xlarge")
model = model.to(f"cuda:{cuda}")
model.eval()

def compute_wer(predictions, references):
    incorrect = 0
    total = 0
    totalS, totalD, totalI = 0, 0, 0
    for prediction, reference in zip(predictions, references):
        measures = compute_measures(reference, prediction)
        H, S, D, I = measures["hits"], measures["substitutions"], measures["deletions"], measures["insertions"]
        totalS += S
        totalD += D
        totalI += I
        incorrect += S + D + I
        total += S + D + H
    
    return {
        "wer": incorrect / total,
        "n_words": total,
        "n_incorrections": incorrect,
        "n_substitutions": totalS,
        "n_deletions": totalD,
        "n_insertions": totalI,
    }


def process_one(path):
    transcription = model.transcribe([path])

    transcription = processor.tokenizer._normalize(transcription[0][0])

    return transcription

idx2gpath_path = Path(generated_prefix, 'idx2gpath', 'all.ptdict')
idx2gpath = torch.load(idx2gpath_path)
sid2idx = {v.stem: k for k, v in idx2gpath.items()}

ref_path = Path("hard_100.txt")
with open(ref_path, 'r') as f:
    references = []
    for i, line in enumerate(f):
        txt = line.strip()
        references.append(txt)

loaded_references = {}
speech_wer_list = []

for generated_folder_name in generated_folder_names:
    generated_paths = Path(generated_folder_name).rglob("*.gen") #NOTELINE:
    generated_paths = list(generated_paths)

    progress = Progress(
        SpinnerColumn(),
        *Progress.get_default_columns(),
        "Elapsed:", TimeElapsedColumn(),
        MofNCompleteColumn(),
    )
    task = progress.add_task("Audio loop", total=len(generated_paths))
    
    with progress:
        for idx, generated_path in enumerate(generated_paths):
            progress.update(task, advance=1)

            rel_path = generated_path.relative_to(generated_prefix)

            speech_id = generated_path.stem

            gen_idx = sid2idx[speech_id]
            reference = processor.tokenizer._normalize(references[gen_idx])

            transcription = process_one(str(generated_path))

            wer_info = compute_wer(references=[reference], predictions=[transcription])
            wer_i = wer_info['wer']
            W_i = wer_info['n_words']
            Inc_i = wer_info['n_incorrections']
            S_i = wer_info['n_substitutions']
            D_i = wer_info['n_deletions']
            Ins_i = wer_info['n_insertions']
            speech_wer_list.append( (gen_idx, rel_path, wer_i, W_i, Inc_i, S_i, D_i, Ins_i) )
            
    progress.stop()

sorted_speech_wer_list = sorted(speech_wer_list, key=lambda x: x[2], reverse=True)
n_words = sum(i[3] for i in sorted_speech_wer_list)
n_incorrections = sum(i[4] for i in sorted_speech_wer_list)
n_substitutions = sum(i[5] for i in sorted_speech_wer_list)
n_deletions = sum(i[6] for i in sorted_speech_wer_list)
n_insertions = sum(i[7] for i in sorted_speech_wer_list)

wer = 100 * n_incorrections / n_words
wsr = 100 * n_substitutions / n_words
wdr = 100 * n_deletions / n_words
wir = 100 * n_insertions / n_words

print("Source: ", generated_prefix)
print("# of sentences: ", len(sorted_speech_wer_list))
print(f'wer: {wer:.4f}, wsr: {wsr:.4f}, wdr: {wdr:.4f}, wir: {wir:.4f}')

results_prefix = Path(f'{generated_prefix}/results')
if not results_prefix.exists():
    results_prefix.mkdir(parents=True)

with open(Path(results_prefix, 'asr.txt'), 'w', encoding='utf-8') as f:
    print(f'wer: {wer:.4f}, wsr: {wsr:.4f}, wdr: {wdr:.4f}, wir: {wir:.4f}', file=f)
    print('rel_path, wer, n_words, n_incorrections, n_substitutions, n_deletions, n_insertions', file=f)
    for gen_idx, speech_rel_path, wer_i, W_i, Inc_i, S_i, D_i, Ins_i in sorted_speech_wer_list:
        print(f"{gen_idx}\t{speech_rel_path}\t{wer_i:.4f}\t{W_i}\t{Inc_i}\t{S_i}\t{D_i}\t{Ins_i}", file=f)