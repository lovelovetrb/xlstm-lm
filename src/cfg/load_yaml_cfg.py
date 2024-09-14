from dataclasses import dataclass

from omegaconf import OmegaConf

# @dataclass
# class ExperimentConfig:
#     pass


def load_config(cfg_path: str):
    print("Loading config...")
    cfg = OmegaConf.load(cfg_path)
    # TODO: dataclassを用いた型チェックの定義
    # OmegaConf.structured(ExperimentConfig)
    return cfg
