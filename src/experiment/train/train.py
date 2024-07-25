import torch
import torch.optim as optim
from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from transformers import set_seed
from xlstm import xLSTMLMModel, xLSTMLMModelConfig

from src.cfg.load_yaml_cfg import load_config
from src.dataset.nlp_dataset import NlpDatasetGenerator
from src.experiment.train.lr_scheduler import LinearWarmupCosineAnnealing
from src.experiment.train.trainer import Trainer
from src.utils import torch_dtype_map, wandb_init


def main():
    config = load_config("src/cfg/yaml/train_config_v2.yaml")
    wandb_init(config)
    set_seed(config.basic.seed)

    dataset = NlpDatasetGenerator(config)
    print("Dataset loaded! ")
    print(f"Train dataset size: {len(dataset.train)}")
    train_loader = DataLoader(
        dataset.train, batch_size=config.training.batch_size, drop_last=True
    )
    pad_token_id = config.dataset.pad_token_id

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
    )

    lr_scheduler = LinearWarmupCosineAnnealing(
        optimizer=optimizer,
        warmup_steps=config.training.lr_warmup_steps,
        decay_until_step=config.training.lr_decay_until_steps,
        max_lr=config.training.lr,
        min_lr=config.training.lr_decay_factor * config.training.lr,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)

    trainer = Trainer(model, train_loader, lr_scheduler, optimizer, criterion, config)
    trainer.train()
    print("Training finished!")


if __name__ == "__main__":
    main()
