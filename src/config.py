import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    model_name: str
    resize_token_embeddings: bool = True
    add_special_tokens: Dict[str, Optional[str]] = field(default_factory=dict)

@dataclass
class TrainConfig:
    seed: int = 42
    output_dir: str = "outputs"
    logging_steps: int = 50
    save_steps: int = 1000
    save_total_limit: int = 3
    eval_steps: int = 500
    evaluation_strategy: str = "steps"
    warmup_steps: int = 200
    max_train_steps: Optional[int] = None
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    per_device_eval_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_eps: float = 1e-8
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 1.0
    fp16: bool = False
    bf16: bool = False
    block_size: int = 512
    shuffle_buffer: int = 10000
    eval_fraction: float = 0.05
    push_to_hub: bool = False
    report_to: str = "none"
    use_gradient_checkpointing: bool = False
    mlflow_autolog: bool = True

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_model_config(path: str) -> ModelConfig:
    return ModelConfig(**load_yaml(path))

def load_train_config(path: str) -> TrainConfig:
    return TrainConfig(**load_yaml(path))
