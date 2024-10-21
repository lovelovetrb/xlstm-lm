from datasets import load_dataset

from src.cfg.config_type import ExperimentConfig


class JaCCDataset:
    def __init__(self, cfg: ExperimentConfig, subset: str) -> None:
        self.cfg = cfg
        self.data = self._load_cc100(subset=subset)

    def _load_cc100(self, subset: str) -> list[dict]:
        return load_dataset(
            "cc100",
            lang="ja",
            split=subset,
            trust_remote_code=True,
            streaming=True,
        )
