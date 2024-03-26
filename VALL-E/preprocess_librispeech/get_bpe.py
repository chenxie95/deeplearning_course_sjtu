import os
from pathlib import Path

import sentencepiece as spm
import torch
from tqdm import tqdm

bpe_model = spm.SentencePieceProcessor(model_file='preprocess_librispeech/train960_size2000.model')

read_prefix = "/mnt/lustre/sjtu/shared/data/asr/rawdata"
write_prefix = "/data/syk"
#folder_names = [f'{read_prefix}/LibriSpeech/train-clean-100', f'{read_prefix}/LibriSpeech/train-clean-360', f'{read_prefix}/LibriSpeech/train-other-500']
#folder_names = [f'{read_prefix}/LibriSpeech/dev-clean', f'{read_prefix}/LibriSpeech/dev-other']
folder_names = [f'{read_prefix}/LibriSpeech/test-clean', f'{read_prefix}/LibriSpeech/test-other']

for folder_name in folder_names:
    transcript_file_paths = Path(folder_name).rglob("*.trans.txt")

    for i in tqdm(list(transcript_file_paths)):
        transcript_file_path = str(i)
        rel_path_parent = str(i.parent.relative_to(read_prefix))
        path_parent = os.path.join(write_prefix, rel_path_parent)
        if not os.path.exists(path_parent):
            os.makedirs(path_parent)

        with open(transcript_file_path, 'r', encoding="utf-8") as f:
            for line in f:
                first_space_index = line.find(' ')
                text_of__line = line[first_space_index + 1: ]

                bpe_code_filename = line[ :first_space_index]
                bpe_code_path = path_parent + '/' + bpe_code_filename + '.bpe'

                bpe_code = bpe_model.encode(text_of__line)
                bpe_code = torch.tensor(bpe_code)
                torch.save(bpe_code, bpe_code_path)

# ! 