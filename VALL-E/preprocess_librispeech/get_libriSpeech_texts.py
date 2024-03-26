from pathlib import Path

from tqdm import tqdm

prefix = "/mnt/lustre/sjtu/shared/data/asr/rawdata/LibriSpeech"
folder_names = [f'{prefix}/train-clean-100', f'{prefix}/train-clean-360', f'{prefix}/train-other-500']

output_file_name = 'preprocess_librispeech/train960_trans.txt'
output_file = open(output_file_name, 'w', encoding='utf-8')

for folder_name in folder_names:
    transcript_file_paths = Path(folder_name).rglob("*.trans.txt")

    for i in tqdm(list(transcript_file_paths)):
        path = str(i)

        with open(path, 'r', encoding="utf-8") as f:
            for line in f:
                first_space_index = line.find(' ')
                text_of_the_line = line[first_space_index + 1: ]
                output_file.write(text_of_the_line)

output_file.close()