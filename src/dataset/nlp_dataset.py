from torch.utils.data import Dataset

from src.dataset.ja_wiki_dataset import JaWikiDataset
from src.dataset.slim_pajama_dataset import SlimPajamaDataset


class NlpDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class NlpDatasetGenerator:
    # TODO: cfgの型を定義
    def __init__(self, cfg):
        self.cfg = cfg
        self.subset = cfg.dataset.subset
        self.datasets = {}
        self._load_data(cfg.dataset.name)

    def _load_data(self, dataset_name: str):
        if dataset_name == "slim_pajama":
            for subset in self.subset:
                # TODO: loggingに変更
                print(
                    f"Loading {subset} dataset from SlimPajama-627B...(this may take a while)"
                )
                slim_pajama_dataset = SlimPajamaDataset(self.cfg, subset=subset)
                self.datasets[subset] = NlpDataset(slim_pajama_dataset.data)
        if dataset_name == "ja_wiki":
            for subset in self.subset:
                print(f"Loading {subset} dataset from  ...(this may take a while)")
                ja_wiki_dataset = JaWikiDataset(self.cfg, subset=subset)
                self.datasets[subset] = NlpDataset(ja_wiki_dataset.data)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")

    @property
    def train(self):
        return self.datasets["train"]

    @property
    def valid(self):
        return self.datasets["valid"]

    @property
    def test(self):
        return self.datasets["test"]
