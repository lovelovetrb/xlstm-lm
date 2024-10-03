import argparse
import functools
import os

import torch
import torch.multiprocessing as mp
import torch.optim as optim
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import set_seed
from xlstm import xLSTMLMModel, xLSTMLMModelConfig

import wandb
from src.cfg.load_yaml_cfg import load_config
from src.dataset.nlp_dataset import NlpDatasetGenerator
from src.experiment.train.lr_scheduler import LinearWarmupCosineAnnealing
from src.experiment.train.trainer_fsdp import TrainerFSDP
from src.utils import dist_cleanup, dist_setup, torch_dtype_map, wandb_init

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(rank, world_size, config):
    wandb_init(config)
    set_seed(config.basic.seed)

    pad_token_id = config.dataset.pad_token_id

    dist_setup(rank, world_size)
    torch.cuda.set_device(rank)

    dataset = NlpDatasetGenerator(config)
    train_sampler = DistributedSampler(
        dataset.train,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    )
    train_loader = DataLoader(
        dataset.train,
        batch_size=config.training.batch_size,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
        sampler=train_sampler,
    )
    if rank == 0:
        print("Dataset loaded! ")
        print(f"Train dataset size: {len(dataset.train)}")

    my_auto_wrap_policy = functools.partial(
        size_based_auto_wrap_policy, min_num_params=10000000
    )
    model_config = from_dict(
        data_class=xLSTMLMModelConfig,
        data=OmegaConf.to_container(config.model, resolve=True),
        config=DaciteConfig(strict=True),
    )
    model = xLSTMLMModel(model_config)
    model.reset_parameters()
    model.train()
    model = model.to(dtype=torch_dtype_map[config.training.weight_precision])
    model = model.to(rank)

    def param_init_fn(module):
        optim_groups = module._create_weight_decay_optim_groups()
        return (
            {"weight_decay": config.training.weight_decay, "params": optim_groups[0]},
            {"weight_decay": 0.0, "params": optim_groups[1]},
        )

    model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
        auto_wrap_policy=my_auto_wrap_policy,
        device_id=torch.cuda.current_device(),
        param_init_fn=param_init_fn
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.training.lr,
        betas=(0.9, 0.95),
        eps=1e-5,
    )

    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer=optimizer,
        warmup_steps=config.training.lr_warmup_steps,
        decay_until_step=config.training.lr_decay_until_steps,
        max_lr=config.training.lr,
        min_lr=config.training.lr_decay_factor * config.training.lr,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

    trainer = TrainerFSDP(
        model, train_loader, lr_scheduler, optimizer, criterion, config, rank
    )
    trainer.train()
    if rank == 0:
        print("Training finished!")
    dist_cleanup()
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    config = load_config(args.config)
    WORLD_SIZE = torch.cuda.device_count()

    mp.spawn(main, args=(WORLD_SIZE, config), nprocs=WORLD_SIZE, join=True)
