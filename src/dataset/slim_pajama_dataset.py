from datasets import load_dataset

from src.cfg.config_type import ExperimentConfig


class SlimPajamaDataset:
    def __init__(self, cfg: ExperimentConfig, subset: str) -> None:
        self.cfg = cfg
        self.data = self._load_slim_pajama(subset)

    # NOTE: This method takes time at first, but it's okay.
    def _load_slim_pajama(self, subset: str) -> list[dict]:
        return load_dataset(
            "cerebras/SlimPajama-627B",
            split=subset,
            streaming=True,
        )
