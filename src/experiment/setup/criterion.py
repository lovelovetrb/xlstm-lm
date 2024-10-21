import torch

from src.cfg.config_type import ExperimentConfig


def setup_criterion(config: ExperimentConfig) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss(ignore_index=config.dataset.pad_token_id)
