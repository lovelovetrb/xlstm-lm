import torch

from src.cfg.config_type import ExperimentConfig


def setup_criterion(config: ExperimentConfig, pad_token_id: int) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
