import math
import os

import torch

import wandb
from tqdm import tqdm

import wandb
from src.utils import torch_dtype_map


class Trainer:
    def __init__(
        self, model, train_loader, lr_scheduler, optimizer, criterion, config, rank
    ):
        print("Trainer initializing...")
        self.model = model
        self.train_loader = train_loader
        self.lr_scheduler = lr_scheduler
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.rank = rank

        self.step = 0
        self.epoch = 1
        self.train_step_num = len(train_loader)
        self.running_loss = torch.tensor(0.0).to(self.rank)
        print(f"Train Step Num: {self.train_step_num}")

    def train(self):
        # TODO: loggingに変更
        monitoring = tqdm(
            self.train_loader, desc=f"#{self.rank:>2} ", position=self.rank + 1
        )
        for _ in range(self.config.training.num_epochs):
            print(f"######### Epoch {self.epoch} #########")
            for batch in monitoring:
                monitoring.set_description_str(
                    f"Rank {self.rank} Steps {self.step+1}/{self.train_step_num} (Loss: {self.running_loss:.4f})"
                )
                self.train_step(batch)
                self.step_logging()
                if self.step % self.config.training.val_every_step == 0:
                    self.valid_step()
                self.step += 1
            self.epoch += 1
            
            wandb.alert(
                title="Epoch Done",
                text=f"Epoch {self.epoch-1} is done! Loss: {self.running_loss:.4f}",
            )

    def train_step(self, batch):
        feature, label = batch["feature"], batch["label"]
        feature, label = feature.to(self.rank), label.to(self.rank)
        with torch.autocast(
            device_type=self.config.basic.device,
            dtype=torch_dtype_map[self.config.training.amp_precision],
            enabled=self.config.training.enable_mixed_precision,
        ):
            self.optimizer.zero_grad()
            outputs = self.model(feature)
            loss = self.criterion(
                outputs.view(-1, self.config.model.vocab_size),
                label.view(-1),
            )
            loss.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            self.running_loss = loss.item()
            if math.isnan(self.running_loss):
                raise ValueError("Loss is NaN!")

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.model.state_dict(), path)
        print(f"Model has been saved to {path}")


    def step_logging(self):
        wandb.log(
            {
                "Loss": self.running_loss,
                "Learning Rate": self.optimizer.param_groups[0]["lr"],
                "Epoch": self.epoch,
                "Step": self.step,
            }
        )

