set -e
# finetune script
python src/f5_tts/train/finetune_cli.py \
    --exp_name F5TTS_v1_Small \
    --finetune \
    --pretrain_path ./ckpts/model_1200000.pt \
    --epochs 1000 \
    --dataset_name sichuan \
    --batch_size_per_gpu 32000 \
    --learning_rate 1e-5