from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.cfg.config_type import ExperimentConfig
from src.dataset.nlp_dataset import (
    DistributedIterableWrapper,
    NlpDataset,
    NlpDatasetGenerator,
)


def get_dataset_generator(config: ExperimentConfig, tokenizer: AutoTokenizer) -> NlpDatasetGenerator:
    return NlpDatasetGenerator(config, tokenizer)


def setup_dataloader(
    dataset_generator: NlpDatasetGenerator, config: ExperimentConfig, rank: int, world_size: int, subset: str
) -> DataLoader:
    dataset = get_dataset_subset(dataset_generator, subset)
    dataset = DistributedIterableWrapper(dataset, world_size, rank)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )


def get_dataset_subset(data: NlpDatasetGenerator, subset: str) -> NlpDataset:
    if subset == "train":
        return data.train
    elif subset == "valid":
        return data.valid
    elif subset == "test":
        return data.test
    else:
        raise ValueError(subset)
