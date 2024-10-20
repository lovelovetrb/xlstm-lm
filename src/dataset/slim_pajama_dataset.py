from datasets import load_dataset
from transformers import AutoTokenizer


class SlimPajamaDataset:
    def __init__(self, cfg, subset):
        self.cfg = cfg
        self.data = self._load_slim_pajama(subset)

    # NOTE: This method takes time at first, but it's okay.
    def _load_slim_pajama(self, subset) -> list[dict]:
        row_text_ds = load_dataset(
            "cerebras/SlimPajama-627B",
            split=subset,
            streaming=True,
        )
        return row_text_ds
