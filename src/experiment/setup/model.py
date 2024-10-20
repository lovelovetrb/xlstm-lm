import torch

from src.cfg.config_type import ExperimentConfig
from src.model.xlstm_model_wrapper import xlstm_model


def setup_model(config: ExperimentConfig, rank: int) -> torch.nn.Module:
    model = xlstm_model(config, rank).get_model()
    return model
