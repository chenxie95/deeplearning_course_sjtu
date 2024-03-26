from pathlib import Path
from shutil import copytree, ignore_patterns

from tqdm import tqdm

read_prefix = "/mnt/lustre/sjtu/shared/data/asr/rawdata"
write_prefix = "/data/syk/MFA"
folder_names = [f'{read_prefix}/LibriSpeech/train-clean-100', f'{read_prefix}/LibriSpeech/train-clean-360', f'{read_prefix}/LibriSpeech/train-other-500']
folder_names += [f'{read_prefix}/LibriSpeech/dev-clean', f'{read_prefix}/LibriSpeech/dev-other']
folder_names += [f'{read_prefix}/LibriSpeech/test-clean', f'{read_prefix}/LibriSpeech/test-other']

for folder_name in folder_names:
    transcript_file_paths = Path(folder_name).rglob("*.trans.txt")
    transcript_file_paths = list(transcript_file_paths)

    for transcript_file_path in tqdm(transcript_file_paths):
        chapter_path = transcript_file_path.parent
        speaker_path = chapter_path.parent
        dataset_split_path = speaker_path.parent
        dataset_location_path = dataset_split_path.parent

        speaker = speaker_path.stem
        dataset_split = dataset_split_path.stem.split('-')[0]

        rel_path_ds_location = dataset_location_path.relative_to(read_prefix)
        write_path_parent = Path(write_prefix, rel_path_ds_location, dataset_split, speaker)
        if not write_path_parent.exists():
            write_path_parent.mkdir(parents=True)

        with open(transcript_file_path, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.split(maxsplit=1)
                sentence_id = line[0]
                text = line[1].strip()

                lab_path = Path(write_path_parent, f"{sentence_id}.lab")

                with open(lab_path, 'w', encoding="utf-8") as f_w:
                    print(text, end='', file=f_w)

        copytree(chapter_path, write_path_parent, ignore=ignore_patterns('*.trans.txt'),dirs_exist_ok=True)

# ! 