from datasets import load_dataset

from src.cfg.config_type import ExperimentConfig


class JaWikiDataset:
    def __init__(self, cfg: ExperimentConfig, subset: str) -> None:
        self.cfg = cfg
        self.data = self._load_ja_wiki(subset=subset)

    def _load_ja_wiki(self, subset: str) -> list[dict]:
        # https://huggingface.co/datasets/fujiki/wiki40b_ja
        return load_dataset(
            "wikimedia/wikipedia",
            "20231101.ja",
            split=subset,
            streaming=True,
        )
