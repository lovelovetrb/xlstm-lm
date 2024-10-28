import torch

from src.cfg.config_type import ExperimentConfig


def setup_criterion(config: ExperimentConfig) -> torch.nn.Module:
    # TODO: ignore_indexをtokenizerから取得するように変更
    # https://github.com/lovelovetrb/xlstm-lm/issues/31
    return torch.nn.CrossEntropyLoss(ignore_index=config.dataset.pad_token_id)
