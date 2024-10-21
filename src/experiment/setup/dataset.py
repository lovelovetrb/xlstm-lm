import torch
from torch.utils.data import DataLoader

from src.cfg.config_type import ExperimentConfig
from src.dataset.nlp_dataset import DistributedIterableWrapper, NlpDatasetGenerator


def setup_dataset(config: ExperimentConfig, rank: int, world_size: int, subset: str) -> DataLoader:
    dataset = NlpDatasetGenerator(config)[subset]
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
