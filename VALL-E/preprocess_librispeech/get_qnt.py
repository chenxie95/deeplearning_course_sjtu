import os
from pathlib import Path

import soundfile
import torch
import torchaudio
from einops import rearrange
from encodec import EncodecModel
from encodec.utils import convert_audio
from tqdm import tqdm

cuda = 1

@torch.inference_mode()
def decode(codes):
    assert codes.dim() == 3
    return model.decode([(codes, None)]), model.sample_rate

def decode_to_file(resps, path: Path):
    assert resps.dim() == 2, f"Require shape (t q), but got {resps.shape}."
    resps = rearrange(resps, "t q -> 1 q t")
    wavs, sr = decode(resps)
    soundfile.write(str(path), wavs.cpu()[0, 0], sr)


@torch.inference_mode()
def encode(wav, sr, device):
    wav = convert_audio(wav, sr, model.sample_rate, model.channels)
    wav = wav.unsqueeze(0)
    wav = wav.to(device)
    encoded_frames = model.encode(wav)
    qnt = torch.cat([encoded[0] for encoded in encoded_frames], dim=-1)  # (b q t)
    return qnt

def encode_from_file(path, device="cuda"):
    wav, sr = torchaudio.load(str(path))
    return encode(wav, sr, device)

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)
model.to(f'cuda:{cuda}')

read_prefix = "/mnt/lustre/sjtu/shared/data/asr/rawdata"
write_prefix = "/data/syk"
#folder_names = [f'{read_prefix}/LibriSpeech/train-clean-100', f'{read_prefix}/LibriSpeech/train-clean-360', f'{read_prefix}/LibriSpeech/train-other-500']
#folder_names = [f'{read_prefix}/LibriSpeech/dev-clean', f'{read_prefix}/LibriSpeech/dev-other']
folder_names = [f'{read_prefix}/LibriSpeech/test-clean', f'{read_prefix}/LibriSpeech/test-other']

for folder_name in folder_names:
    audio_paths = Path(folder_name).rglob("*.flac")

    for audio_path in tqdm(list(audio_paths)):
        rel_path = audio_path.relative_to(read_prefix).with_suffix('.qnt')
        out_path = Path(write_prefix, rel_path)

        if not os.path.exists(out_path.parent):
            os.makedirs(out_path.parent)

        qnt = encode_from_file(audio_path, f'cuda:{cuda}')[0]
        torch.save(qnt.cpu(), out_path)