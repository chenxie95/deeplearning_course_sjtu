import json
from pathlib import Path

import tgt
import torch

from tqdm import tqdm

FREQ = 75

read_prefix = "/data/syk/MFA/aligned/arpa/LibriSpeech"
write_prefix = "/data/syk/LibriSpeech/forced_alignment"
folder_names = [f'{read_prefix}/train', f'{read_prefix}/dev', f'{read_prefix}/test']
dict_path = f"{read_prefix}/phones.dict"

non_speech_annotations = ("sil", "") # SEE: https://montreal-forced-aligner.readthedocs.io/en/latest/user_guide/dictionary.html#non-speech-annotations
unk_annotations = ("spn", "sp")

stats = {'n_filtered': {}, 'n_raw': {}}

with open(dict_path, 'r') as f:
    phone_dict = json.load(f)

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

        phoneme_code_path = (path_parent / textgrid_file_path.stem).with_suffix('.phn')
        textgrid = tgt.io.read_textgrid(textgrid_file_path.as_posix())

        def get_phones(textgrid):

            phones = []

            for t in textgrid.get_tier_by_name("phones")._objects:
                s, e, p = t.start_time, t.end_time, t.text

                if p in non_speech_annotations:
                    continue
                
                if p in unk_annotations:
                    return None

                p = phone_dict[p]

                if len(phones) > 0:
                    last_s, last_e, last_p = phones[-1]
                    if p == last_p and last_e == s:
                        phones[-1] = (last_s, e, p)
                        continue

                phones.append( (s,e,p) )
            
            return phones
        
        phones = get_phones(textgrid)

        if phones is None:
            stats['n_filtered'][folder_name] += 1
            continue

        phoneme_code = {
            's': torch.tensor([int(s*FREQ) for s,_,_ in phones]),
            'e': torch.tensor([int(e*FREQ) for _,e,_ in phones]),
            'p': torch.tensor([p for _,_,p in phones]),
        }

        torch.save(phoneme_code, phoneme_code_path)

with open(Path(write_prefix, 'filtered.stats'), "w") as f:
    print("n_filtered: ", file=f)
    for folder_name, n in stats['n_filtered'].items():
        print(folder_name, n, file=f)

    print(file=f)
    print("n_raw: ", file=f)
    for folder_name, n in stats['n_raw'].items():
        print(folder_name, n, file=f)


# ! 