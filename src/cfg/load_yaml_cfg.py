from dacite import from_dict
from omegaconf import OmegaConf

from src.cfg.config_type import ExperimentConfig


def load_config(cfg_path: str) -> ExperimentConfig:
    cfg = OmegaConf.load(cfg_path)
    return from_dict(data_class=ExperimentConfig, data=OmegaConf.to_container(cfg, resolve=True))
