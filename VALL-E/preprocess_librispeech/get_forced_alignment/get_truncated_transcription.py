import json
from pathlib import Path

import tgt
import torch

from tqdm import tqdm

FREQ = 75

read_prefix = "/home/ubuntu/MFA/aligned/arpa/LibriSpeech"
write_prefix = "/home/ubuntu/forced_alignment"
folder_names = [f'{read_prefix}/train', f'{read_prefix}/dev', f'{read_prefix}/test']

non_speech_annotations = ("sil", "") # SEE: https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html#non-speech-annotations
unk_annotations = ("spn", "sp")

stats = {'n_filtered': {}, 'n_raw': {}}

for folder_name in folder_names:
    textgrid_file_paths = Path(folder_name).rglob("*.TextGrid")
    textgrid_file_paths = list(textgrid_file_paths)
    stats['n_filtered'][folder_name] = 0
    stats['n_raw'][folder_name] = len(textgrid_file_paths)

    for textgrid_file_path in tqdm(textgrid_file_paths):
        rel_path_parent = textgrid_file_path.parent.relative_to(read_prefix)
        path_parent = Path(write_prefix, rel_path_parent)
        if not path_parent.exists():
            path_parent.mkdir(parents=True)

        transcription_path = (path_parent / textgrid_file_path.stem).with_suffix('.trans')
        textgrid = tgt.io.read_textgrid(textgrid_file_path.as_posix())

        def get_trans(textgrid):

            words = []

            for t in textgrid.get_tier_by_name("words")._objects:
                s, e, p = t.start_time, t.end_time, t.text

                if s < 3.0:
                    continue

                if p in non_speech_annotations:
                    continue
                
                if p in unk_annotations:
                    return None

                words.append( p )
            
            if len(words) == 0:
                return None

            return words
        
        trans = get_trans(textgrid)

        if trans is None:
            stats['n_filtered'][folder_name] += 1
            continue

        with open(transcription_path, 'w', encoding='utf-8') as f_trans:
            print_txt = ' '.join(trans)
            print(print_txt, file=f_trans)

with open(Path(write_prefix, 'filtered.stats'), "w") as f:
    print("n_filtered: ", file=f)
    for folder_name, n in stats['n_filtered'].items():
        print(folder_name, n, file=f)

    print(file=f)
    print("n_raw: ", file=f)
    for folder_name, n in stats['n_raw'].items():
        print(folder_name, n, file=f)