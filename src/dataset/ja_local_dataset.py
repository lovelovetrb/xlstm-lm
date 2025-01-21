from datasets import load_dataset

from src.cfg.config_type import ExperimentConfig


class JaLocalDataset:
    def __init__(self, cfg: ExperimentConfig, subset: str) -> None:
        self.data = self._load_data(file_path=cfg.dataset.name, subset=subset)

    @staticmethod
    def process_data(x: dict) -> dict:
        return {"text": x["text"]}

    def _load_data(self, file_path: str, subset: str) -> list[dict]:
        dataset = load_dataset(
            "text",
            data_files={f"{subset}": [f"src/dataset/data/{file_path}"]},
            streaming=True,
        )
        return dataset[subset].map(self.process_data)
