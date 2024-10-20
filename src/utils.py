import datetime
import logging
import os

import torch
import torch.distributed as dist

import wandb

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def is_serializable(value):
    return isinstance(value, (int, float, str, bool))


def wandb_init(config):
    config_serializable = {key: value for key, value in config.items() if is_serializable(value)}
    wandb.init(
        project=config.basic.project_name,
        name=f"{config.dataset.name}-{config.basic.project_tag}-{config.training.lr}",
        config=config_serializable,
    )


def dist_setup(rank, world_size):
    try:
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(minutes=30),
        )
        torch.cuda.set_device(rank)
        print(f"Rank {dist.get_rank()}: Process group initialized")
    except Exception as e:
        print(f"Failed to initialize process group: {e}")
        raise


def dist_cleanup():
    dist.destroy_process_group()


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    return logger
