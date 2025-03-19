# F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching

This is a minimal PyTorch implementation of the [F5-TTS](https://arxiv.org/abs/2410.06885), changed from [official Github](https://github.com/SWivid/F5-TTS).

## Installation

### Create a separate environment with conda

```bash
conda create -n f5-tts python=3.10
conda activate f5-tts
```

Then download the version of [pytorch](https://pytorch.org/get-started/previous-versions/) that matches your OS and cuda version. Here is an example.
```bash
# Install PyTorch
pip install torch==2.4.0+cu124 torchaudio==2.4.0+cu124 --extra-index-url https://download.pytorch.org/whl/cu124
```
### Create a separate environment with docker image

```bash
# Build from Dockerfile
docker build -t f5tts:v1 .

# Or pull from GitHub Container Registry
docker pull ghcr.io/swivid/f5-tts:main
```

### Install F5-TTS
You can install F5-TTS from this repository with local editable mode.

```bash
git clone https://github.com/chenxie95/deeplearning_course_sjtu.git
cd F5-TTS
# git submodule update --init --recursive  # (optional, if need > bigvgan)
pip install -e .
```

## Inference

### 1. Gradio App

Currently supported features:

- Basic TTS with Chunk Inference
- Multi-Style / Multi-Speaker Generation
- Voice Chat powered by Qwen2.5-3B-Instruct
- [Custom inference with more language support](src/f5_tts/infer/SHARED.md)

```bash
# Launch a Gradio app (web interface)
f5-tts_infer-gradio

# Specify the port/host
f5-tts_infer-gradio --port 7860 --host 0.0.0.0

# Launch a share link
f5-tts_infer-gradio --share
```

### 2. CLI Inference

```bash
# Run with flags
# Leave --ref_text "" will have ASR model transcribe (extra GPU memory usage)
f5-tts_infer-cli --model F5TTS_v1_Base \
--ref_audio "provide_prompt_wav_path_here.wav" \
--ref_text "The content, subtitle or transcription of reference audio." \
--gen_text "Some text you want TTS model generate for you."

# Run with default setting. src/f5_tts/infer/examples/basic/basic.toml
f5-tts_infer-cli
# Or with your own .toml file
f5-tts_infer-cli -c custom.toml

# Multi voice. See src/f5_tts/infer/README.md
f5-tts_infer-cli -c src/f5_tts/infer/examples/multi/story.toml
```

### 3. More instructions

- In order to have better generation results, take a moment to read [detailed guidance](src/f5_tts/infer).
- The [Issues](https://github.com/SWivid/F5-TTS/issues?q=is%3Aissue) are very useful, please try to find the solution by properly searching the keywords of problem encountered. If no answer found, then feel free to open an issue.

## Training & Finetuning

We will provide a pretrained F5-TTS Small model for this course to finetune.
This model is trained on the [Emilia dataset](https://huggingface.co/datasets/amphion/Emilia-Dataset/tree/fc71e07) and can be downloaded from [here](). If you want to train your own model or finetune this model, please read the following guidance for more instructions on how to prepare the dataset and train the model.

### Prepare Dataset

Example data processing scripts, and you may tailor your own one along with a Dataset class in `src/f5_tts/model/dataset.py`.

We provide some specific Datasets preparing scripts for you to start with. Download corresponding dataset first, and fill in the path in scripts.

```bash
# Prepare the Emilia dataset
python src/f5_tts/train/datasets/prepare_emilia.py

# Prepare the Wenetspeech4TTS dataset
python src/f5_tts/train/datasets/prepare_wenetspeech4tts.py

# Prepare the LibriTTS dataset
python src/f5_tts/train/datasets/prepare_libritts.py

# Prepare the LJSpeech dataset
python src/f5_tts/train/datasets/prepare_ljspeech.py
```
if you want to create custom dataset with metadata.csv, you can use guidance see [#57 here](https://github.com/SWivid/F5-TTS/discussions/57#discussioncomment-10959029).

```bash
python src/f5_tts/train/datasets/prepare_csv_wavs.py
```

### Training & Finetuning

Once your datasets are prepared, you can start the training process.

#### 1. Training script used for pretrained model

```bash
# setup accelerate config, e.g. use multi-gpu ddp, fp16
# will be to: ~/.cache/huggingface/accelerate/default_config.yaml     
accelerate config

# .yaml files are under src/f5_tts/configs directory
accelerate launch src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml

# possible to overwrite accelerate and hydra config
accelerate launch --mixed_precision=fp16 src/f5_tts/train/train.py --config-name F5TTS_v1_Base.yaml ++datasets.batch_size_per_gpu=19200
```

#### 2. Finetuning practice
Discussion board for Finetuning [#57](https://github.com/SWivid/F5-TTS/discussions/57).

Gradio UI training/finetuning with `src/f5_tts/train/finetune_gradio.py` see [#143](https://github.com/SWivid/F5-TTS/discussions/143).

The `use_ema = True` is harmful for early-stage finetuned checkpoints (which goes just few updates, thus ema weights still dominated by pretrained ones), try turn it off and see if provide better results.

#### 3. W&B Logging
if you want to use W&B logging, you can use the following command to run the training script.
The `wandb/` dir will be created under path you run training/finetuning scripts.

```bash
# wandb login
export WANDB_API_KEY=your_api_key
# or you can use `wandb login` to login

# if your computing resource is limited to access the internet
export WANDB_MODE=offline

# if you want to sync the offline log files to the cloud
wandb sync wandb/
```

#### 4. With Gradio App
if you want to use Gradio web interface to train/finetune, you can use the following command to run the training script.
```bash
# Quick start with Gradio web interface
f5-tts_finetune-gradio
```

## Evaluation

Install evaluation packages:
```bash
pip install -e .[eval]
```
Detailed evaluation instructions are available in the [evaluation](src/f5_tts/eval).
## More Infroamtion
If you want to know more about the project, please refer to the [official Github](https://github.com/SWivid/F5-TTS). There are more detailed instructions and explanations in [issues](https://github.com/SWivid/F5-TTS/issues) and [discussions](https://github.com/SWivid/F5-TTS/discussions). If you have any questions, please feel free to open an issue in official Github or contact your TA.

If you think this project is helpful to you, please cite the following paper:
```
@article{chen-etal-2024-f5tts,
      title={F5-TTS: A Fairytaler that Fakes Fluent and Faithful Speech with Flow Matching}, 
      author={Yushen Chen and Zhikang Niu and Ziyang Ma and Keqi Deng and Chunhui Wang and Jian Zhao and Kai Yu and Xie Chen},
      journal={arXiv preprint arXiv:2410.06885},
      year={2024},
}
```
## License

Our code is released under MIT License. The pre-trained models are licensed under the CC-BY-NC license due to the training data Emilia, which is an in-the-wild dataset. Sorry for any inconvenience this may cause.
