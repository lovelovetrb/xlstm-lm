import argparse
import os

import torch
import torch.multiprocessing as mp
from transformers import set_seed

import wandb
from src.cfg.config_type import ExperimentConfig
from src.cfg.load_yaml_cfg import load_config
from src.experiment.setup.criterion import setup_criterion
from src.experiment.setup.dataloader import get_dataset_generator, setup_dataloader
from src.experiment.setup.lr_scheduler import setup_lr_scheduler
from src.experiment.setup.model import setup_model
from src.experiment.setup.optimizer import setup_optimizer
from src.experiment.setup.tokenizer import setup_tokenizer
from src.experiment.train.trainer import Trainer, TrainerArgs
from src.utils import dist_cleanup, dist_setup, get_logger, wandb_init

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def setup_training(rank: int, world_size: int, config: ExperimentConfig) -> TrainerArgs:
    logger = get_logger(f"train-{rank}")
    logger.info(f"Rank {rank}: Setting up training...")

    wandb_init(rank, config)
    set_seed(config.basic.seed)
    dist_setup(rank, world_size, logger)

    tokenizer = setup_tokenizer(config.tokenizer.name)
    dataset_generator = get_dataset_generator(config, tokenizer)
    train_loader = setup_dataloader(dataset_generator, config, rank, world_size, "train")
    model = setup_model(config, rank)
    optimizer = setup_optimizer(model, config)
    lr_scheduler = setup_lr_scheduler(optimizer, config)
    criterion = setup_criterion(config, tokenizer.pad_token_id)

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


def main(rank: int, world_size: int, config: ExperimentConfig) -> None:
    try:
        trainer_args = setup_training(rank, world_size, config)
        trainer = Trainer(trainer_args)
        trainer.train()
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

    mp.spawn(main, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
