This is the PyTorch implementation of the VALLE paper implemented by [@Ereboas](https://github.com/Ereboas). This folder is based on the [fairseq package v0.12.2](https://github.com/pytorch/fairseq/tree/v0.12.2).

# Requirements and Installation

It's recommended to use conda to create environments and install packages. 
``` bash
conda create -n valle python=3.9
conda activate valle
```

If you need to install conda first, see [Appendix A](#a-conda-installation).

Then download the version of [pytorch](https://pytorch.org/get-started/previous-versions/) that matches your OS and cuda version. Here is an example.
``` bash
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu117
```

Finally, install valle from this repository.
``` bash
pip install -e .
```

Other dependencies:
``` bash
pip install -e git+https://git@github.com/facebookresearch/encodec#egg=encodec
pip install transformers
pip install jiwer
pip install libsndfile1
pip install ffmpeg
pip install soundfile
pip install Cython
pip install nemo_toolkit['all']==1.20.0
pip install tensorboardX
pip install pyarrow
```

Don't hesitate to contact TA if you encounter any installation problems.

# Training

## Data Preparing
After extracting valle_data.tar, you will get two folders: `forced_alignment`, and `LibriSpeech`. Note that `${data_split_path}` of the following commands is the absolute path of `forced_alignment/splits`. You may modify the first two lines of `forced_alignment/splits/train|valid|test.tsv` to match the absolute path.

## AR model
``` bash
#WANDB_NAME=valle \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
fairseq-train ${data_split_path} \
--save-dir checkpoints/valle \
#--restore-file checkpoints/valle/checkpoint10.pt \
#--wandb-project valle \
--task language_modeling --modified -a transformer_lm \
--skip-invalid-size-inputs-valid-test --skip-remainder-batch --max-target-positions 4096 \
--optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --clip-norm 0.0 \
--lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 32000 --warmup-init-lr 1e-07 \
--tokens-per-sample 4096 --max-tokens 4096 --update-freq 4 \
--fp16 --max-update 200000 --num-workers 3 \
--ar-task --text-full-attention \
--n-control-symbols 80
```

## NAR model
``` bash
#WANDB_NAME=valle-nar \
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
fairseq-train ${data_split_path} \
--save-dir checkpoints/valle-nar \
#--wandb-project valle-nar \
--task language_modeling --modified -a transformer_lm \
--skip-invalid-size-inputs-valid-test --skip-remainder-batch --max-target-positions 4096 \
--optimizer adam --adam-betas '(0.9,0.98)' --weight-decay 0.01 --clip-norm 0.0 \
--lr 0.0005 --lr-scheduler inverse_sqrt --warmup-updates 32000 --warmup-init-lr 1e-07 \
--tokens-per-sample 4096 --max-tokens 4096 --update-freq 3 \
--fp16 --max-update 200000 --num-workers 3 \
--n-control-symbols 80
```

## Speech Synthesis
``` bash
python eval_code/generate_all.py
```

## Speech Evaluation
``` bash
python eval_code/asr.py
```

# Appendix

The appendix mainly provides some additional information. In most cases, this section **does not** need to be read.

## A. Conda installation
Most compute resources already have conda installed by default. If your environment requires manual conda installation, please refer to this link: 
https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html

# License

This repository is under MIT license.
