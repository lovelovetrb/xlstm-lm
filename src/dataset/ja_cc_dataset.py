import torch
from datasets import load_dataset
from transformers import AutoTokenizer


class JaCCDataset:
    def __init__(self, cfg, subset):
        self.cfg = cfg
        self.data = self._load_cc100(subset=subset)

    def _load_cc100(self, subset) -> list[dict]:
        if subset == "train":
            ratio = self.cfg.dataset.train_ratio
        elif subset == "valid":
            ratio = self.cfg.dataset.valid_ratio
        elif subset == "test":
            ratio = self.cfg.dataset.test_ratio
        else:
            raise ValueError(f"Subset {subset} not supported")
        row_text_ds = load_dataset(
            "cc100",
            lang="ja",
            split=subset,
            # num_proc=self.num_proc,
            trust_remote_code=True,
            streaming=True,
        )
        return row_text_ds
