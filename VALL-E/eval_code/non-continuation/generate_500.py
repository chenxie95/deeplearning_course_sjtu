from pathlib import Path

import torch
import torchaudio
from torch.distributions import Categorical
from fairseq.models.transformer_lm import TransformerLanguageModel
from encodec import EncodecModel

from rich.progress import Progress
from rich.progress import SpinnerColumn, TimeElapsedColumn, MofNCompleteColumn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--cuda_id', '-c', type=int, default=1, help='ID of the CUDA device to use')
parser.add_argument('--split_id', '-s', type=int, default=-1)
parser.add_argument('--max_split_id', '-m', type=int, default=4)
parser.add_argument('--sample_p', '-p', type=float, default=0.99)
parser.add_argument('--runs', '-r', type=int, default=3) #! set to -1 to unable multirun.
parser.add_argument('--begin_run_idx', '-b', type=int, default=0)
args = parser.parse_args()

MUL_CONST = 75
PROP_SEC = 3
TOO_SILENT_SEC = 2
TOO_SHORT_SEC = 4
TOO_LONG_SEC = 10
GEN_SEC_MAX = 40
EOS = 2
PAD = 1
BOS = 0
SAMPLE_P = args.sample_p
TEMPERATURE = 1

cuda = args.cuda_id
split = args.cuda_id if args.split_id == -1 else args.split_id
MAXSPLIT = args.max_split_id
runs = args.runs
arcp = 30
narcp = 40

qnt_prefix = "/home/ubuntu/LibriSpeech"
phn_prefix = "/home/ubuntu/forced_alignment"
audio_prefix = phn_prefix
model_name = f"valle-phone"
experiment_name = f"{model_name}-ar{arcp}-nar{narcp}-p{SAMPLE_P}-t{TEMPERATURE}"
generate_prefix = f"./generated_non-continuation/{experiment_name}" if args.runs == -1 else f"./generated_non-continuation/{experiment_name}"

ar_path = "./valle-phone"
nar_path = "./valle-phone-nar"
audio_folder_names = [f'{audio_prefix}/test']

encodec_model = EncodecModel.encodec_model_24khz()
encodec_model.set_target_bandwidth(6.0)
encodec_model = encodec_model.to(f"cuda:{cuda}")

ar_model = TransformerLanguageModel.from_pretrained(ar_path, f'checkpoint{arcp}.pt')
ar_model = ar_model.models[0].to(f"cuda:{cuda}")
nar_model = TransformerLanguageModel.from_pretrained(nar_path, f'checkpoint{narcp}.pt')
nar_model = nar_model.models[0].to(f"cuda:{cuda}")
encodec_model.eval()
ar_model.eval()
nar_model.eval()

phone_path = "hard_100_tokens.txt"
with open(phone_path, 'r') as f:
    f_lines = f.readlines()
    gen_p_list = []
    for i in range(len(f_lines)):
        ps = list(map(int, f_lines[i].split()))
        gen_p_list.append(torch.tensor(ps))

begin_idx = int(split * len(gen_p_list) // MAXSPLIT)
end_idx = int((split+1) * len(gen_p_list) // MAXSPLIT)

def top_p_sampling(x, *, p):
    x_sorted, indices = torch.sort(x, descending=True)
    cumulative_probs = torch.cumsum(x_sorted, axis=-1)
    mask = cumulative_probs <= p
    mask[0] = True

    top_p_probs = x_sorted[mask]
    top_p_indices = indices[mask]

    top_p_probs /= top_p_probs.sum()
    prob = torch.zeros_like(x)
    prob[top_p_indices] = top_p_probs

    dist = Categorical(prob)
    output = dist.sample()

    return output

def top_k_sampling(x, *, k):
    top_k_probs, top_k_indices = torch.topk(x, k=k)
    prob = torch.zeros_like(x)
    prob[top_k_indices] = top_k_probs / top_k_probs.sum()

    dist = Categorical(prob)
    output = dist.sample()

    return output

def generate_one(phn_path, qnt_path):
    phn_code_dict = torch.load(phn_path)
    s, e, p = phn_code_dict['s'], phn_code_dict['e'], phn_code_dict['p']
    pp = gen_p_list[to_gen_idx]
    if ((s > MUL_CONST*PROP_SEC).sum())==0:return "Short"
    ii = (s > MUL_CONST*PROP_SEC).nonzero()[0,0]
    p = torch.cat((p[:ii], pp))
    ee=torch.cat((e[:ii], s[ii]+10*torch.arange(len(pp))+10))
    ss=torch.cat((s[:ii], s[ii]+10*torch.arange(len(pp))))
    s = ss
    e = ee
    p = p + 4 + 1024
    qnt_code = torch.load(qnt_path) + 4
    qnt_code = torch.cat((qnt_code, torch.zeros((8, 10*len(pp)), dtype=torch.int64)), dim=1)

    if qnt_code.size(1) < MUL_CONST * TOO_SHORT_SEC:
        return "Short"

    if qnt_code.size(1) > MUL_CONST * (TOO_LONG_SEC+len(pp)):
        return "Long"
    
    if s[0] > MUL_CONST * TOO_SILENT_SEC:
        return "Silent start"

    if s[-1] < MUL_CONST * PROP_SEC:
        return "Early end"

    prompt_length = MUL_CONST * PROP_SEC

    phn_column_code = p.repeat(8, 1)
    data_source = torch.cat([
        phn_column_code,
        torch.LongTensor([BOS]).repeat(8, 1),
        qnt_code[:, :prompt_length],
    ], dim=1).to(f"cuda:{cuda}")

    ar_source = data_source[0]

    net_input = {
        "src_tokens": ar_source[None],
        "src_lengths": torch.LongTensor([ar_source.numel()]),
        "separators": torch.LongTensor([len(p) + 1]),
    }

    with torch.no_grad():
        i, max_iter = 0, MUL_CONST*GEN_SEC_MAX
        while i < max_iter:
            progress.update(task2, advance=1)
            net_output = ar_model(**net_input)
            lprob = ar_model.get_normalized_probs(net_output, log_probs=False)[0, -1]
            lprob = lprob[:1024+4]
            lprob[BOS] = 0
            lprob[PAD] = 0
            lprob = lprob / lprob.sum()
            output_i = top_p_sampling(lprob, p=SAMPLE_P)
            ar_source = torch.cat([ar_source, output_i[None]])
            net_input = {
                "src_tokens": ar_source[None],
                "src_lengths": torch.LongTensor([ar_source.numel()]),
                "separators": torch.LongTensor([len(p) + 1]),
            }
            if output_i == EOS:
                break
            
            i = i + 1

        nar_source = [ar_source[None]]
        for j in range(1, 8):
            net_input = {
                "src_tokens": nar_source,
                "src_lengths": torch.LongTensor([ar_source.numel()]),
                "separators": torch.LongTensor([len(p) + 1]),
                "train_nar_layer": j,
            }
            output_i = nar_model(**net_input)[0][0, :, 4:1024+4].argmax(-1) + 4
            output_i[:prompt_length+len(p)+1] = data_source[j, :prompt_length+len(p)+1]
            nar_source.append(output_i[None])

    whole_sentence = torch.cat(nar_source)[:, len(p)+1:]

    if whole_sentence[0, -1] == EOS:
        whole_sentence = whole_sentence[:, :-1]
    
    whole_sentence -= 4

    with torch.no_grad():
        whole_wav = encodec_model.decode([
            (whole_sentence[None], None)
        ])[0].cpu()
    
    gen_wav = whole_wav[:, PROP_SEC*encodec_model.sample_rate:]

    if gen_wav.size(1) <= encodec_model.sample_rate * 0.01:
        return "Short generation"
    

    return whole_sentence.cpu(), whole_wav, gen_wav

if runs == -1:
    runs = 1
    args.begin_run_idx = 0

for run in range(args.begin_run_idx, args.begin_run_idx+runs):
    generate_prefix_per_run = Path(generate_prefix, f'run{run}')
    if args.runs == -1:
        generate_prefix_per_run = generate_prefix

    n_bad_sentences = 0
    n_valid_sentences = 0
    counter_details = {}
    idx2gpath = {}
    to_gen_idx = begin_idx
    print(generate_prefix_per_run, 'Start')
    for audio_folder_name in audio_folder_names:
        if to_gen_idx == end_idx: break
        audio_paths = Path(audio_folder_name).rglob("*.phn")
        audio_paths = list(audio_paths)
        _ap = []
        for audio_path in audio_paths:
            speaker, chapter = audio_path.stem.split('-')[:2]
            qnt_path = next(Path(qnt_prefix).glob(f"*/{speaker}")) / chapter / audio_path.with_suffix(".qnt").name
            if 'test-clean' in qnt_path.parts: _ap.append(audio_path)
        audio_paths = _ap
        len_audio_paths = len(audio_paths)
        audio_paths = audio_paths[int(split * len_audio_paths // MAXSPLIT): int((split+1) * len_audio_paths // MAXSPLIT)]

        progress = Progress(
            SpinnerColumn(),
            *Progress.get_default_columns(),
            "Elapsed:", TimeElapsedColumn(),
            MofNCompleteColumn(),
        )
        task1 = progress.add_task(f"Run{run} Audio loop", total=len(audio_paths))
        task2 = progress.add_task("Generate loop", total=MUL_CONST*GEN_SEC_MAX)
        
        with progress:
            for idx, audio_path in enumerate(audio_paths):
                if to_gen_idx == end_idx: break
                progress.update(task1, advance=1)

                rel_path = audio_path.relative_to(audio_prefix)
                phn_path = Path(phn_prefix, rel_path.with_suffix('.phn'))
                speaker, chapter = phn_path.stem.split('-')[:2]
                qnt_path = next(Path(qnt_prefix).glob(f"*/{speaker}")) / chapter / phn_path.with_suffix(".qnt").name
                if 'test-other' in qnt_path.parts: continue
                whole_out_path = Path(generate_prefix_per_run, qnt_path.relative_to(qnt_prefix).with_suffix('.whole'))
                gen_out_path = Path(generate_prefix_per_run, qnt_path.relative_to(qnt_prefix).with_suffix('.gen'))
                qnt_out_path = Path(generate_prefix_per_run, qnt_path.relative_to(qnt_prefix).with_suffix('.qnt'))

                if not whole_out_path.parent.exists():
                    whole_out_path.parent.mkdir(parents=True)

                generate_one_return = generate_one(phn_path, qnt_path)  #* 24kHz
                if generate_one_return == "Short":
                    progress.console.print(f"{rel_path} is too short (<{TOO_SHORT_SEC}s), skipped.")
                    continue
                elif generate_one_return == "Long":
                    progress.console.print(f"{rel_path} is too long (>{TOO_LONG_SEC}s), skipped.")
                    continue
                elif generate_one_return == "Silent start":
                    progress.console.print(f"{rel_path} starts with too much silence (>{TOO_SILENT_SEC}s), skipped.")
                    continue
                elif generate_one_return == "Early end":
                    progress.console.print(f"{rel_path} the last phone starts too early (<{PROP_SEC}s), skipped.")
                    continue
                elif generate_one_return == "Short generation":
                    progress.console.print(f"{rel_path} generate too shortly (<{0.01}s), maybe because of a silent tail, skipped.")
                    continue
                
                idx2gpath[to_gen_idx] = gen_out_path
                to_gen_idx = to_gen_idx + 1

                qnt_out, whole_out, gen_out = generate_one_return
                
                this_speech_bad = bool(qnt_out.size(1) >= (MUL_CONST*GEN_SEC_MAX-5))
                if this_speech_bad:
                    n_bad_sentences += 1
                
                n_valid_sentences += 1

                counter_details[str(whole_out_path)] = (int(this_speech_bad), 1)

                gen_out_16k = torchaudio.transforms.Resample(encodec_model.sample_rate, 16000)(gen_out)
                torchaudio.save(gen_out_path, gen_out_16k, 16000, format="wav")
                
                whole_out_16k = torchaudio.transforms.Resample(encodec_model.sample_rate, 16000)(whole_out)
                torchaudio.save(whole_out_path, whole_out_16k, 16000, format="wav")

                torch.save(qnt_out, qnt_out_path)
                #orig_wav, orig_sr = torchaudio.load(str(audio_path)) # ! check orig_sr = 16000
                #assert orig_sr == 16000

                progress.reset(task2)
        
        progress.stop()

    counter_path = Path(generate_prefix_per_run, 'bad_counter', f"split{split}.counter")
    if not counter_path.parent.exists():
        counter_path.parent.mkdir(parents=True)

    idx2gpath_path = Path(generate_prefix_per_run, 'idx2gpath', f"split{split}.ptdict")
    if not idx2gpath_path.parent.exists():
        idx2gpath_path.parent.mkdir(parents=True)
    torch.save(idx2gpath, idx2gpath_path)

    counter_save = {
        'valid_sentences': n_valid_sentences, 'bad_sentences': n_bad_sentences
    }
    torch.save(counter_save, counter_path)

    counter_details_path = Path(generate_prefix_per_run, 'bad_counter', f"split{split}.details")
    sorted_counter_details_list = sorted(counter_details.items(), key=lambda x: -x[1][0]/x[1][1])
    with open(counter_details_path, 'w', encoding='utf-8') as f:
        print(f"Run {run}. Split{split}/{MAXSPLIT} total sentences: {n_valid_sentences}, bad sentences: {n_bad_sentences}, approx: {n_bad_sentences/n_valid_sentences:.4f}", file=f)
        for speech_path, (counter_i, total_i) in sorted_counter_details_list:
            print(f"{speech_path}\t{counter_i}\t{total_i}", file=f)

    print(generate_prefix_per_run, 'Done')
    print(f"Run {run}. Split{split}/{MAXSPLIT} total sentences: {n_valid_sentences}, bad sentences: {n_bad_sentences}, approx: {n_bad_sentences/n_valid_sentences:.4f}")