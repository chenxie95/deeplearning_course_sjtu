import json
from pathlib import Path

import tgt
from tqdm import tqdm

read_prefix = "/data/syk/MFA/aligned/arpa/LibriSpeech"
folder_names = [f'{read_prefix}/train', f'{read_prefix}/dev', f'{read_prefix}/test']

phone_set = set()

for folder_name in folder_names:
    textgrid_file_paths = Path(folder_name).rglob("*.TextGrid")
    textgrid_file_paths = list(textgrid_file_paths)

    for textgrid_file_path in tqdm(textgrid_file_paths):
        textgrid = tgt.io.read_textgrid(textgrid_file_path.as_posix())

        for t in textgrid.get_tier_by_name("phones")._objects:
            s, e, p = t.start_time, t.end_time, t.text
            phone_set.add(t.text)

phone_dict = {value: index for index, value in enumerate(sorted(phone_set))}

dict_save_path = Path(read_prefix, 'phones.dict')
with open(dict_save_path, "w") as f:
    json.dump(phone_dict, f)


# ! 