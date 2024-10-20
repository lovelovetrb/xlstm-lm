import argparse
import logging
import os
from typing import Tuple

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import set_seed

import wandb
from src.cfg.config_type import ExperimentConfig
from src.cfg.load_yaml_cfg import load_config
from src.experiment.setup.criterion import setup_criterion
from src.experiment.setup.dataset import setup_dataset
from src.experiment.setup.lr_scheduler import setup_lr_scheduler
from src.experiment.setup.model import setup_model
from src.experiment.setup.optimizer import setup_optimizer
from src.experiment.train.trainer import Trainer
from src.utils import dist_cleanup, dist_setup, get_logger, wandb_init

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_training(
    rank: int, world_size: int, config: ExperimentConfig, logger: logging.Logger
) -> Tuple[torch.nn.Module, DataLoader, optim.Optimizer, torch.nn.Module, torch.nn.Module]:
    logger.info(f"Rank {rank}: Setting up training...")
    wandb_init(config)
    set_seed(config.basic.seed)
    dist_setup(rank, world_size, logger)

    train_loader = setup_dataset(config, rank, world_size)
    model = setup_model(config, rank)
    optimizer = setup_optimizer(model, config)
    lr_scheduler = setup_lr_scheduler(optimizer, config)
    criterion = setup_criterion(config)

    return model, train_loader, lr_scheduler, optimizer, criterion


def cleanup() -> None:
    dist_cleanup()
    wandb.finish()


def main(rank: int, world_size: int, config: ExperimentConfig, logger: logging.Logger) -> None:
    try:
        model, train_loader, lr_scheduler, optimizer, criterion = setup_training(rank, world_size, config, logger)
        trainer = Trainer(model, train_loader, lr_scheduler, optimizer, criterion, config, rank, logger)
        trainer.train()
    except Exception as e:
        logger.error(f"Error in main: {e}")
    finally:
        cleanup()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()

    logger = get_logger(__name__)
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(main, args=(WORLD_SIZE, config, logger), nprocs=WORLD_SIZE, join=True)
