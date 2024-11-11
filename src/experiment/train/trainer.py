import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

import wandb
from src.cfg.config_type import ExperimentConfig
from src.utils import get_logger, torch_dtype_map


@dataclass
class TrainerArgs:
    model: torch.nn.Module
    train_loader: torch.utils.data.DataLoader
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler
    optimizer: torch.optim.Optimizer
    criterion: torch.nn.Module
    config: ExperimentConfig
    rank: int


class Trainer:
    def __init__(self, args: TrainerArgs) -> None:
        self.logger = get_logger(f"Trainer: {args.rank}")
        if args.rank == 0:
            self.logger.info("Trainer initializing...")
        self.model = args.model
        self.train_loader = args.train_loader
        self.lr_scheduler = args.lr_scheduler
        self.optimizer = args.optimizer
        self.criterion = args.criterion
        self.config = args.config
        self.rank = args.rank
        self.step = 0
        self.epoch = 1
        self.running_loss = torch.tensor(0.0).to(self.rank)

        # checkpoint保存用のディレクトリを作成
        if args.rank == 0:
            Path(self.config.training.model_save_dir).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Model save directory: {self.config.training.model_save_dir} created.")
        dist.barrier()

    def train(self) -> None:
        for _ in range(self.config.training.num_epochs):
            self._train_epoch()

    def _train_epoch(self) -> None:
        if self.rank == 0:
            self.logger.info(f"######### Epoch {self.epoch} #########")
        for batch in self._get_progress_bar():
            self._train_step(batch)
            self._step_logging()
            self._validate_if_needed()
            self.step += 1
        self._epoch_logging()
        self.epoch += 1

    def _step_logging(self) -> None:
        wandb.log(
            {
                "Loss": self.running_loss,
                "Learning Rate": self.optimizer.param_groups[0]["lr"],
                "Epoch": self.epoch,
                "Step": self.step,
            }
        )

    def _epoch_logging(self) -> None:
        wandb.alert(
            title="Epoch Done",
            text=f"Epoch {self.epoch - 1} is done! Loss: {self.running_loss}",
        )

    def _get_progress_bar(self) -> tqdm:
        return tqdm(
            self.train_loader,
            desc=f"Epoch {self.epoch}",
            position=self.rank + 1,
            leave=False,
            dynamic_ncols=True,
        )

    def _train_step(self, batch: dict) -> None:
        feature, label = batch["feature"].to(self.rank), batch["label"].to(self.rank)
        with torch.autocast(
            device_type=self.config.basic.device,
            dtype=torch_dtype_map[self.config.training.amp_precision],
            enabled=self.config.training.enable_mixed_precision,
        ):
            self._clear_cache()
            self.optimizer.zero_grad()
            outputs = self.model(feature)
            loss = self._compute_loss(outputs, label)
            self._backward_and_optimize(loss)
            self._update_running_loss(loss)
            dist.barrier()

    def _clear_cache(self) -> None:
        if self.config.basic.device == "cuda":
            torch.cuda.empty_cache()

    def _compute_loss(self, outputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        return self.criterion(
            outputs.view(-1, self.config.model.vocab_size),
            label.view(-1),
        )

    def _backward_and_optimize(self, loss: torch.Tensor) -> None:
        loss.backward()
        self.optimizer.step()
        self.lr_scheduler.step()

    def _update_running_loss(self, loss: torch.Tensor) -> None:
        self.running_loss = loss.item()
        self._check_nan_loss()

    def _check_nan_loss(self) -> None:
        if math.isnan(self.running_loss):
            self.logger.error("NaN Loss detected!")
            raise ValueError(self.running_loss)

    def _validate_if_needed(self) -> None:
        if self.step % self.config.training.val_every_step == 0 and self.step != 0:
            self._valid()
            self.save_model_wrapper(f"{self.config.training.model_save_dir}/model_{self.step}.pth")

    def _valid(self) -> None:
        self.model.eval()
        self.valid_loss = 0.0
        valid_steps = self.config.training.val_steps
        self._valid_step(valid_steps)

    def _valid_step(self, valid_steps: int) -> None:
        for batch in self.train_loader:
            feature, label = batch["feature"].to(self.rank), batch["label"].to(self.rank)
            with torch.no_grad():
                outputs = self.model(feature)
                loss = self._compute_loss(outputs, label)
                self.valid_loss += loss.item()
                valid_steps -= 1
                if valid_steps == 0:
                    break
        self.valid_loss /= self.config.training.val_steps

    def _valid_step_logging(self) -> None:
        self.logger.info(f"Validation Loss: {self.valid_loss} | steps: {self.step}")
        wandb.log(
            {
                "Validation Loss": self.valid_loss,
                "Epoch": self.epoch,
                "Step": self.step,
            }
        )

    def text_generate(self) -> None:
        pass

    def save_model_wrapper(self, save_path: str) -> None:
        dist.barrier()
        if self.rank == 0:
            self.logger.info(f"checkpoint saving : {save_path}")
        if self.config.training.use_fsdp:
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(
                    rank0_only=True,
                ),
            ):
                if self.rank == 0:
                    self.save_model(save_path)
        else:
            if self.rank == 0:
                self.save_model(save_path)

        dist.barrier()
        self.model.train()

    def save_model(self, save_path: str) -> None:
        state_dict = self.model.state_dict()
        torch.save(state_dict, save_path)
        self.logger.info(f"Model has been saved to {save_path}")
