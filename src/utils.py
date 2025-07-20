import os, random, torch, numpy as np
from pathlib import Path

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def count_trainable_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def human_readable(n: int) -> str:
    for unit in ["", "K", "M", "B"]:
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else str(n)
        n /= 1000
    return f"{n:.1f}T"
