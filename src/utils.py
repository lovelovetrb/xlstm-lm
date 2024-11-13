import datetime
import sys
import typing

import torch
import torch.distributed as dist
import wandb
from loguru import logger

from src.cfg.config_type import ExperimentConfig

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


# ruff: noqa: ANN401
def is_serializable(value: typing.Any) -> bool:
    return isinstance(value, (int, float, str, bool))


def wandb_init(rank: int, config: ExperimentConfig) -> None:
    # config_serializable = {
    #     key: value for key, value in config.items() if is_serializable(value)
    # }
    wandb.init(
        project=config.basic.project_name,
        group=f"{config.dataset.name}-{config.basic.project_tag}-{config.training.lr}",
        name=f"RUN: {rank}",
        config=config,
    )


def dist_setup(rank: int, world_size: int, logger: logger) -> None:
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=30),
        )
        torch.cuda.set_device(rank)
        logger.info(f"Rank {dist.get_rank()}: Process group initialized")
    except Exception as e:
        logger.exception(f"Failed to initialize process group: {e}")
        raise


def dist_cleanup() -> None:
    dist.destroy_process_group()


def get_logger(name: str) -> logger:
    logger.add(sys.stderr, format="{time} {level} {message}", filter=name, level="INFO")
    return logger
