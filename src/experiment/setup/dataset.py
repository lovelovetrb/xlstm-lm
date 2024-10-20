import torch
from torch.utils.data import DataLoader

from src.cfg.config_type import ExperimentConfig
from src.dataset.nlp_dataset import DistributedIterableWrapper, NlpDatasetGenerator


def setup_dataset(config: ExperimentConfig, rank: int, world_size: int) -> DataLoader:
    dataset = NlpDatasetGenerator(config)
    distributed_dataset = DistributedIterableWrapper(dataset.train, world_size, rank)
    train_loader = DataLoader(
        distributed_dataset,
        batch_size=config.training.batch_size,
        drop_last=True,
        num_workers=torch.cuda.device_count(),
        pin_memory=True,
        shuffle=False,
    )
    return train_loader
