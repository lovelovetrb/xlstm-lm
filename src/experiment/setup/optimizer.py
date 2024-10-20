import torch
import torch.optim as optim

from src.cfg.config_type import ExperimentConfig


def setup_optimizer(model: torch.nn.Module, config: ExperimentConfig) -> optim.Optimizer:
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        betas=(0.9, 0.95),
        eps=1e-5,
    )
    return optimizer
