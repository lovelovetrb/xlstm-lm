import os

import torch
import torch.multiprocessing as mp
import wandb
from transformers import set_seed

from src.cfg.config_type import ExperimentConfig
from src.cfg.load_yaml_cfg import load_config
from src.dataset.nlp_dataset import NlpDatasetGenerator
from src.experiment.train.lr_scheduler import LinearWarmupCosineAnnealing
from src.experiment.train.trainer import Trainer

from src.utils import dist_cleanup, dist_setup, torch_dtype_map, wandb_init

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main(config):
    wandb_init(config)
    set_seed(config.basic.seed)
    dataset = NlpDatasetGenerator(config)
    train_loader = DataLoader(
        dataset.train,
        batch_size=config.training.batch_size,
        drop_last=True,
        num_workers=1,
        pin_memory=True,
        shuffle=False,
    )
    print("Dataset loaded! ")
    print(f"Train dataset size: {len(dataset.train)}")

    model_config = from_dict(
        data_class=xLSTMLMModelConfig,
        data=OmegaConf.to_container(config.model, resolve=True),
        config=DaciteConfig(strict=True),
    )
    model = xLSTMLMModel(model_config)
    model.reset_parameters()
    model.train()
    model = model.to(dtype=torch_dtype_map[config.training.weight_precision])
    model = model.to(config.basic.device)

    optim_groups = model._create_weight_decay_optim_groups()

    optimizer = optim.AdamW(
        (
            {"weight_decay": config.training.weight_decay, "params": optim_groups[0]},
            {"weight_decay": 0.0, "params": optim_groups[1]},
        ),
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

    trainer = Trainer(
        model, train_loader, lr_scheduler, optimizer, criterion, config, rank=0
    )
    trainer.train()
    print("Training finished!")
    wandb.finish()


if __name__ == "__main__":
    config = load_config("src/cfg/yaml/v4/train_config.yaml")
    try:
        main(config)
    except Exception as e:
        print(e)
