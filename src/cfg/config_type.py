from dataclasses import dataclass, field
from typing import List

from xlstm import xLSTMLMModelConfig


@dataclass
class BasicConfig:
    seed: int
    device: str
    project_name: str
    project_tag: str
    mode: str
    model_weight_path: str


@dataclass
class DatasetConfig:
    name: str
    min_seq_length: int
    max_seq_length: int
    subset: List[str] = field(default_factory=list)


@dataclass
class TokenizerConfig:
    name: str


@dataclass
class TrainingConfig:
    num_epochs: int
    batch_size: int
    valid_step: int
    use_fsdp: bool
    lr: float
    val_every_step: int
    lr_warmup_steps: int
    lr_decay_until_steps: int
    lr_decay_factor: float
    weight_decay: float
    num_steps: int
    amp_precision: str
    weight_precision: str
    enable_mixed_precision: bool
    model_save_dir: str


@dataclass
class ExperimentConfig:
    basic: BasicConfig
    dataset: DatasetConfig
    tokenizer: TokenizerConfig
    model: xLSTMLMModelConfig
    training: TrainingConfig
