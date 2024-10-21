import argparse
import logging
import os

import torch
import torch.multiprocessing as mp
import wandb
from transformers import set_seed

from src.cfg.config_type import ExperimentConfig
from src.cfg.load_yaml_cfg import load_config
from src.experiment.setup.criterion import setup_criterion
from src.experiment.setup.dataset import setup_dataset
from src.experiment.setup.lr_scheduler import setup_lr_scheduler
from src.experiment.setup.model import setup_model
from src.experiment.setup.optimizer import setup_optimizer
from src.experiment.train.trainer import Trainer, TrainerArgs
from src.utils import dist_cleanup, dist_setup, get_logger, wandb_init

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_training(rank: int, world_size: int, config: ExperimentConfig, logger: logging.Logger) -> TrainerArgs:
    logger.info(f"Rank {rank}: Setting up training...")
    wandb_init(config)
    set_seed(config.basic.seed)
    dist_setup(rank, world_size, logger)

    train_loader = setup_dataset(config, rank, world_size, "train")
    model = setup_model(config, rank)
    optimizer = setup_optimizer(model, config)
    lr_scheduler = setup_lr_scheduler(optimizer, config)
    criterion = setup_criterion(config)

    return TrainerArgs(
        model=model,
        train_loader=train_loader,
        lr_scheduler=lr_scheduler,
        optimizer=optimizer,
        criterion=criterion,
        config=config,
        rank=rank,
    )


def cleanup() -> None:
    dist_cleanup()
    wandb.finish()


def main(rank: int, world_size: int, config: ExperimentConfig, logger: logging.Logger) -> None:
    try:
        trainer_args = setup_training(rank, world_size, config, logger)
        trainer = Trainer(trainer_args)
        trainer.train()
    except Exception as e:
        logger.exception(f"Error in main: {e}")
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
