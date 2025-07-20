#!/usr/bin/env bash
set -e

export TOKENIZERS_PARALLELISM=false

python -m src.entry_train \
  --model_config config/model_config.yaml \
  --train_config config/train_config.yaml \
  --data_dir data/raw \
  --pattern shakespeare.txt
