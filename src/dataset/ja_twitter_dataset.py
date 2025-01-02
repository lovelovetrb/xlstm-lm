from datasets import load_dataset

from src.cfg.config_type import ExperimentConfig


class JaLocalDataset:
    def __init__(self, cfg: ExperimentConfig, subset: str) -> None:
        self.cfg = cfg
        self.data = self._load_ja_wiki(subset=subset)

    def _load_ja_wiki(self, file_path: str) -> list[dict]:
        return load_dataset(
            "text",
            data_dir=f"./data/{file_path}",
            streaming=True,
        )
