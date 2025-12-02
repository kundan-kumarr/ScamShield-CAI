import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class BaseConfig:
    base_model_name: str
    revision_model_name: str
    sft_output_dir: str
    dpo_output_dir: str
    max_new_tokens: int
    temperature: float
    top_p: float
    train_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    num_train_epochs: int


def load_config(path: str = "configs/base_config.json") -> BaseConfig:
    with open(path, "r") as f:
        cfg = json.load(f)
    return BaseConfig(**cfg)
