import torch

from src.cfg.config_type import ExperimentConfig
from src.model.xlstm_model_wrapper import xLSTMModelWrapper


# TODO: checkpointの読み込み処理
# issue: https://github.com/lovelovetrb/xlstm-lm/issues/25
def setup_model(config: ExperimentConfig, rank: int, model_weight_path: str | None = None) -> torch.nn.Module:
    return xLSTMModelWrapper(config, rank, model_weight_path).get_model()
