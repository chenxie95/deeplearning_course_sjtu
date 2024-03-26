import argparse
import glob
import os
import random

import torch

from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", metavar="DIR", help="root directory containing folders of audio files to index",
        default="/data/syk/LibriSpeech"
    )
    parser.add_argument(
        "--audio_folders", metavar="DIR", nargs="+", help="audio directories under the root",
        default=["train-clean-100", "train-clean-360", "train-other-500"]
    )
    parser.add_argument(
        "--valid-percent",
        default=0.02,
        type=float,
        metavar="D",
        help="percentage of data to use as validation set (between 0 and 1)",
    )
    parser.add_argument(
        "--dest", default="/data/syk/Vall_E_data", type=str, metavar="DIR", help="output directory"
    )
    parser.add_argument(
        "--ext", default="bpe", type=str, metavar="EXT", help="extension to look for"
    )
    parser.add_argument("--seed", default=42, type=int, metavar="N", help="random seed")
    parser.add_argument(
        "--path-must-contain",
        default=None,
        type=str,
        metavar="FRAG",
        help="if set, path must contain this substring for a file to be included in the manifest",
    )
    return parser


def main(args):
    assert args.valid_percent >= 0 and args.valid_percent <= 1.0

    if not os.path.exists(args.dest):
        os.makedirs(args.dest)

    rand = random.Random(args.seed)

    valid_f = (
        open(os.path.join(args.dest, "valid.tsv"), "w")
        if args.valid_percent > 0
        else None
    )
    
    with open(os.path.join(args.dest, "train.tsv"), "w") as train_f:
        
        root_path = os.path.realpath(args.root)
        print(root_path, file=train_f)

        if valid_f is not None:
            print(root_path, file=valid_f)

        for audio_folder_path in args.audio_folders:

            dir_path = os.path.join(root_path, audio_folder_path)
            search_path = os.path.join(dir_path, "**/*." + args.ext)


            for fname in tqdm(list(glob.iglob(search_path, recursive=True))):
                file_path = os.path.realpath(fname)

                if args.path_must_contain and args.path_must_contain not in file_path:
                    continue
                file_prefix = file_path[ :file_path.rfind('.')]
                qnt_code = torch.load(file_prefix+'.qnt')
                bpe_code = torch.load(file_prefix+'.bpe')
                size = qnt_code.size(1) + bpe_code.size(0) + 2

                #frames = soundfile.info(fname).frames
                dest = train_f if rand.random() > args.valid_percent else valid_f
                print(
                    "{}\t{}".format(os.path.relpath(file_path, root_path), size), file=dest
                )
    
    if valid_f is not None:
        valid_f.close()


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
