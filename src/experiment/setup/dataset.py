import torch
from torch.utils.data import DataLoader

from src.cfg.config_type import ExperimentConfig
from src.dataset.nlp_dataset import (
    DistributedIterableWrapper,
    NlpDataset,
    NlpDatasetGenerator,
)


def setup_dataset(config: ExperimentConfig, rank: int, world_size: int, subset: str) -> DataLoader:
    # TODO: DatasetGeneratorに対してtokenizerを渡すような実装に変更
    dataset_generator = NlpDatasetGenerator(config)
    dataset = get_dataset_subset(dataset_generator, subset)
    if config.training.use_fsdp:
        dataset = DistributedIterableWrapper(dataset, world_size, rank)
    return DataLoader(
        dataset,
        batch_size=config.training.batch_size,
        drop_last=True,
        num_workers=torch.cuda.device_count(),
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
