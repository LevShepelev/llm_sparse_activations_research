from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from pathlib import Path
import math, random

def load_raw_text(path_pattern: str) -> str:
    """Concatenate all .txt files under path_pattern (directory or file)."""
    p = Path(path_pattern)
    if p.is_file():
        return p.read_text(encoding="utf-8")
    texts = []
    for f in sorted(p.glob("*.txt")):
        texts.append(f.read_text(encoding="utf-8"))
    return "\n\n".join(texts)

def build_dataset(tokenizer: PreTrainedTokenizerBase, text: str, block_size: int, eval_fraction: float):
    tokens = tokenizer(text, return_tensors=None)["input_ids"]
    # Drop last partial block
    n_blocks = (len(tokens) // block_size)
    tokens = tokens[: n_blocks * block_size]
    blocks = [tokens[i*block_size:(i+1)*block_size] for i in range(n_blocks)]
    random.shuffle(blocks)
    split = max(1, int(len(blocks) * eval_fraction))
    eval_blocks = blocks[:split]
    train_blocks = blocks[split:]

    def to_ds(block_list):
        return Dataset.from_dict({"input_ids": block_list})

    return to_ds(train_blocks), to_ds(eval_blocks)

def with_format(ds: Dataset):
    return ds.with_format(type="torch")
