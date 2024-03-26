from pathlib import Path

import torch
from tqdm import tqdm

N_SPECIAL_TOKENS = 2
CONTROL_MULTIPLIER = 3
ACOUSTIC_MULTIPLIER = 1
# 一些Train sequence的组织方式
# <PHONEME SEQ> <BOS> <FUSION SEQ> <EOS>: Size = 2 + 2 * PHONEME + ACOUSTIC
# <PHONEME> <DURATION> <ACOUSTIC OF PHONEME> <EOP> ...: SIZE = 3 * PHONEME + ACOUSTIC

phn_prefix = "/home/ubuntu/forced_alignment"
qnt_prefix = "/home/ubuntu/LibriSpeech"
dest_folder = Path(phn_prefix, "splits_231")

if not dest_folder.exists():
    dest_folder.mkdir(parents=True)

splits = ['train', 'dev', 'test']
for split in splits:
    split_fname = Path(dest_folder, f"{split}.tsv")

    with open(split_fname, "w") as f:
        split_root = Path(phn_prefix, split).resolve()
        qnt_root = Path(qnt_prefix).resolve()
        print(split_root, file=f)
        print(qnt_root, file=f)

        phn_paths = split_root.rglob(f'*.phn')
        phn_paths = list(phn_paths)
        for phn_path in tqdm(phn_paths):
            speaker, chapter = phn_path.stem.split('-')[:2]

            qnt_path = next(Path(qnt_prefix).glob(f"*/{speaker}")) / chapter / phn_path.with_suffix(".qnt").name
            if not qnt_path.exists():
                print(phn_path.stem)
                continue
            qnt_code = torch.load(qnt_path)
            phn_code = torch.load(phn_path)

            size = ACOUSTIC_MULTIPLIER * qnt_code.size(1) + \
                   CONTROL_MULTIPLIER * phn_code['p'].size(0) + N_SPECIAL_TOKENS
            print(
                f"{phn_path.relative_to(split_root)}\t{size}", file=f
            )