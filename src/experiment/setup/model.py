import torch

from src.cfg.config_type import ExperimentConfig
from src.model.xlstm_model_wrapper import xLSTMModelWrapper


def setup_model(config: ExperimentConfig, rank: int) -> torch.nn.Module:
    return xLSTMModelWrapper(config, rank).get_model()
