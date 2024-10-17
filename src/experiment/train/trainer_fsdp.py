import math
import os

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from tqdm import tqdm

import wandb
from src.utils import torch_dtype_map


class TrainerFSDP:
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
        self.running_loss = torch.tensor(0.0).to(self.rank)

    def train(self):
        # TODO: loggingに変更
        monitoring = tqdm(
            self.train_loader, desc=f"#{self.rank:>2} ", position=self.rank + 1
        )
        for _ in range(self.config.training.num_epochs):
            if self.rank == 0:
                print(f"######### Epoch {self.epoch} #########")
            for batch in monitoring:
                monitoring.set_description_str(
                    f"Rank {self.rank} Steps {self.step+1} (Loss: {self.running_loss:.4f})"
                )
                self.train_step(batch)
                self.step_logging()
                if self.step % self.config.training.val_every_step == 0:
                    self.valid_step()
                self.step += 1
            self.epoch += 1
            wandb.alert(
                title="Epoch Done",
                text=f"Epoch {self.epoch-1} is done! Loss: {self.running_loss}",
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

            del outputs, loss
            if math.isnan(self.running_loss):
                print(self.running_loss)
                raise ValueError("Loss is NaN!")

    def valid_step(self):
        self.save_model(f"{self.config.training.model_save_dir}/model_{self.step}.pth")

    def save_model(self, save_path):
        # Save checkpoints.
        with FSDP.state_dict_type(
            self.model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(
                rank0_only=True,
                offload_to_cpu=True,
                # Default value is to translate back to Hugging Face Transformers format,
                # when saving full checkpoints for models trained with SMP tensor parallelism.
                # translate_on_save=True
            ),
        ):
            state_dict = self.model.state_dict()
            if dist.get_rank() == 0:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                # This name is needed for HF from_pretrained API to work.
                torch.save(state_dict, save_path)
                print(f"Model has been saved to {save_path}")
            dist.barrier()

    def step_logging(self):
        wandb.log(
            {
                "Loss": self.running_loss,
                "Learning Rate": self.optimizer.param_groups[0]["lr"],
                "Epoch": self.epoch,
                "Step": self.step,
            }
        )
