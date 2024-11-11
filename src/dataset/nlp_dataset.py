import torch
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer

from src.cfg.config_type import ExperimentConfig
from src.dataset.ja_cc_dataset import JaCCDataset
from src.dataset.ja_wiki_dataset import JaWikiDataset
from src.dataset.ntp_data_generator import NextTokenPredictionDataGenerator
from src.dataset.slim_pajama_dataset import SlimPajamaDataset
from src.utils import get_logger


class DistributedIterableWrapper(IterableDataset):
    def __init__(self, dataset: IterableDataset, num_replicas: int, rank: int) -> None:
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # 各ワーカーとランクの組み合わせに対して一意のシードを生成
        seed = self.rank * 10000 + worker_id
        torch.manual_seed(seed)

        # データセットのイテレータを取得
        dataset_iter = iter(self.dataset)

        for i, item in enumerate(dataset_iter):
            # この項目が現在のランクとワーカーに属するかを確認
            if i % (self.num_replicas * num_workers) == self.rank * num_workers + worker_id:
                yield item


class NlpDataset(IterableDataset):
    def __init__(self, data: list[dict], cfg: ExperimentConfig, tokenizer: AutoTokenizer) -> None:
        self.data = data
        self.cfg = cfg
        self.ntp_data_generator = NextTokenPredictionDataGenerator(cfg, tokenizer)

    # ruff: noqa: ANN204
    def __iter__(self):
        for item in self.data:
            tokenized_data = self.ntp_data_generator.tokenize_dataset(item["text"])
            yield tokenized_data


class NlpDatasetGenerator:
    def __init__(self, cfg: ExperimentConfig, tokenizer: AutoTokenizer) -> None:
        self.cfg = cfg
        self.subset = cfg.dataset.subset
        self.datasets: dict[str, NlpDataset] = {}
        self.logger = get_logger("NlpDatasetGenerator")
        self.tokenizer = tokenizer
        self._load_data(cfg.dataset.name)

    def _load_data(self, dataset_name: str) -> None:
        for subset in self.subset:
            if dataset_name == "slim_pajama":
                self.logger.info(f"Loading {subset} dataset from SlimPajama-627B...(this may take a while)")
                slim_pajama_dataset = SlimPajamaDataset(self.cfg, subset=subset)
                self.datasets[subset] = NlpDataset(slim_pajama_dataset.data, self.cfg, self.tokenizer)
            elif dataset_name == "ja_wiki":
                self.logger.info(f"Loading {subset} dataset from ja_wiki_40b ...(this may take a while)")
                ja_wiki_dataset = JaWikiDataset(self.cfg, subset=subset)
                self.datasets[subset] = NlpDataset(ja_wiki_dataset.data, self.cfg, self.tokenizer)
            elif dataset_name == "ja_cc":
                self.logger.info(f"Loading {subset} dataset from ja_cc...(this may take a while)")
                ja_cc_wiki_dataset = JaCCDataset(self.cfg, subset=subset)
                self.datasets[subset] = NlpDataset(ja_cc_wiki_dataset.data, self.cfg, self.tokenizer)
            else:
                self.logger.error(f"Dataset {dataset_name} not supported")
                raise ValueError(dataset_name)

    @property
    def train(self) -> NlpDataset:
        return self.datasets["train"]

    @property
    def valid(self) -> NlpDataset:
        return self.datasets["valid"]

    @property
    def test(self) -> NlpDataset:
        return self.datasets["test"]
