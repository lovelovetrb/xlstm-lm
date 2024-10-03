import torch
import torch.distributed as dist

import wandb

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def wandb_init(config):
    wandb.init(
        project=config.basic.project_name,
        config=config,
    )


def dist_setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def dist_cleanup():
    dist.destroy_process_group()
