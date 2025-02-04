from datasets import load_dataset

from src.cfg.config_type import ExperimentConfig
from src.utils import get_logger


class JaLocalDataset:
    def __init__(self, cfg: ExperimentConfig, subset: str) -> None:
        logger = get_logger("JaLocalDataset")
        logger.info(f"Loading {cfg.dataset.name} dataset")
        self.data = self._load_data(file_path=cfg.dataset.name, subset=subset)

    @staticmethod
    def process_data(x: dict) -> dict:
        return {"text": x["text"]}

    def _load_data(self, file_path: str, subset: str) -> list[dict]:
        dataset = load_dataset(
            "text", data_files={f"{subset}": [f"src/dataset/data/{file_path}"]}, streaming=True, encoding="utf-8"
        )
        return dataset[subset].map(self.process_data)
