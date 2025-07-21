# GPT-2 Small (124 M) Fine-Tuning on Shakespeare  
*Poetry + 🤗 Transformers + MLflow*

<p align="center">
  <img src="https://raw.githubusercontent.com/akutruff/shakespeare-gpt2-assets/main/cover.jpg" width="600" alt="GPT-2 meets Shakespeare">
</p>

---

## 1 · Why this repo?

* **One-command training** of GPT-2-small on any folder of `.txt` files (default: Shakespeare).
* **Poetry** for reproducible env management.
* **MLflow** auto-logs ➜ live curves & artifacts (loss, perplexity plot, checkpoints).
* Tiny, modular code (≤ 400 LOC) that’s easy to swap for other corpora, models, or PEFT methods.

---

## 2 · Project Structure

gpt2_shakespeare/
├── README.md ← you are here
├── pyproject.toml ← Poetry deps & scripts
├── config/
│ ├── model_config.yaml ← hub model + extra tokens
│ ├── train_config.yaml ← hparams, eval/log cadence
│ └── mlflow_config.yaml ← tracking-URI, experiment name
├── data/
│ └── raw/ ← put *.txt here (multiple files OK)
├── outputs/ ← checkpoints & artifacts (auto)
├── src/
│ ├── callbacks.py ← Perplexity + Early-Stop
│ ├── config.py ← dataclass loaders
│ ├── data.py ← dataset builder
│ ├── model.py ← tokenizer / model loader
│ ├── trainer.py ← helpers for TrainingArguments
│ ├── utils.py ← misc utils
│ └── entry_train.py ← main training entry-point
└── scripts/
├── prepare_dataset.py ← optional cleaner
└── entry_generate.py ← sample text after training

bash
Copy
Edit

---

## 3 · Quick Start

```bash
# 1 · Install Poetry (if you don’t have it)
curl -sSL https://install.python-poetry.org | python3 -

# 2 · Clone repo & install deps
git clone https://github.com/your-org/gpt2_shakespeare.git
cd gpt2_shakespeare
poetry install --with dev   # creates virtual-env, installs GPU Torch etc.

# 3 · Add data
mkdir -p data/raw
cp /path/to/your/*.txt data/raw/

# 4 · Launch MLflow UI (optional but nice)
poetry run mlflow ui  --port 5000 &
open http://localhost:5000   # or just visit in browser

# 5 · Train
poetry run python -m src.entry_train \
  --model_config config/model_config.yaml \
  --train_config config/train_config.yaml \
  --data_dir data/raw \
  --pattern "*.txt"
A new MLflow run appears with live curves: train_loss, eval_loss, perplexity, plus a PNG plot when training finishes.

4 · Configuration Cheat-Sheet
File	Key knobs
config/model_config.yaml	model_name (hub id), add_special_tokens, resize_token_embeddings
config/train_config.yaml	num_train_epochs, learning_rate, block_size, per_device_train_batch_size, gradient_accumulation_steps, eval_steps, logging_steps, use_gradient_checkpointing, fp16 / bf16
config/mlflow_config.yaml	tracking_uri (local dir / http), experiment_name, run_name

All YAMLs are parsed into dataclasses → no changes in code needed.

5 · Generate Text
bash
Copy
Edit
poetry run python scripts/entry_generate.py \
  --model_dir outputs/gpt2-shakespeare \
  --prompt "O, gentle night," \
  --max_new_tokens 120 \
  --temperature 0.95 \
  --top_p 0.9
entry_generate.py uses the same tokenizer; supports multiple sampling params.

