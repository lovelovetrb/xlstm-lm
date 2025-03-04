import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from tqdm import tqdm

import wandb
from src.cfg.config_type import ExperimentConfig
from src.experiment.setup.tokenizer import setup_tokenizer
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
        wandb.watch(self.model)
        self.train_loader = args.train_loader
        self.lr_scheduler = args.lr_scheduler
        self.optimizer = args.optimizer
        self.criterion = args.criterion
        self.config = args.config
        self.rank = args.rank
        self.epoch = 1
        self.step = 0
        self.iter = 0
        self.running_loss = torch.tensor(0.0).to(self.rank)
        self.valid_loss = torch.tensor(0.0).to(self.rank)
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.config.training.enable_mixed_precision)

        self.tokenizer = setup_tokenizer(self.config.tokenizer.name)

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
        for batch_idx, batch in self._get_progress_bar():
            self._train_step(batch_idx, batch)
        self._epoch_logging()
        self.epoch += 1

    def _step_logging(self, loss: torch.Tensor) -> None:
        wandb.log(
            {
                "Epoch": self.epoch,
                "Step": self.step,
                "iter": self.iter,
                "Accum Loss": self.running_loss,
                "Last Original Loss": loss.item(),
                "Learning Rate": self.optimizer.param_groups[0]["lr"],
            }
        )

    def _epoch_logging(self) -> None:
        wandb.alert(
            title="Epoch Done",
            text=f"Epoch {self.epoch - 1} is done! Loss: {self.running_loss}",
        )

    def _get_progress_bar(self) -> tqdm:
        return tqdm(
            enumerate(self.train_loader),
            desc=f"Epoch {self.epoch}",
            position=self.rank + 1,
            leave=False,
            dynamic_ncols=True,
        )

    def _train_step(self, batch_idx: int, batch: dict) -> None:
        feature, label = batch["feature"].to(self.rank), batch["label"].to(self.rank)
        self._clear_cache()
        with torch.autocast(
            device_type=self.config.basic.device,
            dtype=torch_dtype_map[self.config.training.amp_precision],
            enabled=self.config.training.enable_mixed_precision,
        ):
            # feature.shape: (batch_size, seq_len)
            outputs = self.model(feature)
            # outputs.shape: (batch_size, seq_len, vocab_size)
            loss = self._compute_loss(outputs, label)
        self._backward(loss)
        self._accumulate_loss(loss)
        self.iter += 1
        if (batch_idx + 1) % self.config.training.grad_accum_steps == 0:
            self._optimize()
            self._update_running_loss()
            self._check_nan_loss()
            self._step_logging(loss)
            self._validate_if_needed()
            self.step += 1

    def _clear_cache(self) -> None:
        if self.config.basic.device == "cuda":
            torch.cuda.empty_cache()

    def _compute_loss(self, outputs: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
        loss = self.criterion(
            outputs.view(-1, self.config.model.vocab_size),
            label.view(-1),
        )
        return loss / self.config.training.grad_accum_steps

    def _backward(self, loss: torch.Tensor) -> None:
        if self.config.training.enable_mixed_precision:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimize(self) -> None:
        if self.config.training.enable_mixed_precision:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
        else:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()

    def _accumulate_loss(self, loss: torch.Tensor) -> None:
        if not hasattr(self, "_accumulated_loss"):
            self._accumulated_loss = 0
        self._accumulated_loss += loss.item() * self.config.training.grad_accum_steps

    def _update_running_loss(self) -> None:
        self.running_loss = self._accumulated_loss
        self._accumulated_loss = 0

    def _check_nan_loss(self) -> None:
        if math.isnan(self.running_loss):
            self.logger.error("NaN Loss detected!")
            raise ValueError(self.running_loss)

    def _validate_if_needed(self) -> None:
        if self.step % self.config.training.val_every_step == 0 and self.step != 0:
            self._valid()
            dist.barrier()

    def _valid(self) -> None:
        self.save_model_wrapper(f"{self.config.training.model_save_dir}/model_{self.step}.pth")
        self.model.eval()
        self.valid_loss = 0.0
        self.train_loader.dataset.set_start_index(self.iter)
        self._valid_step()
        self.train_loader.dataset.set_start_index(self.iter + self.config.training.val_steps)
        self.model.train()

    def _valid_step(self) -> None:
        self.logger.info(f"Step {self.step} Validation...\n")
        for step, batch in self._get_progress_bar():
            feature, label = batch["feature"].to(self.rank), batch["label"].to(self.rank)
            with torch.no_grad():
                outputs = self.model(feature)
                self.text_generate(outputs)
                loss = self._compute_loss(outputs, label)
                loss *= self.config.training.grad_accum_steps
                self.valid_loss += loss.item()
                if step == self.config.training.val_steps:
                    break
        self.valid_loss /= self.config.training.val_steps
        self.valid_ppl = math.exp(self.valid_loss)
        self._valid_step_logging()
        self._clear_cache()
        dist.barrier()

    def _valid_step_logging(self) -> None:
        self.logger.info(f"Step {self.step} Validation -> loss: {self.valid_loss} | ppl: {self.valid_ppl}")
        wandb.log(
            {
                "Epoch": self.epoch,
                "Step": self.step,
                "valid loss": self.valid_loss,
                "valid ppl": self.valid_ppl,
            }
        )

    def text_generate(self, outputs: torch.Tensor) -> None:
        if self.rank == 0:
            self.logger.info(f"model generate text: {self.tokenizer.decode(torch.argmax(outputs, dim=-1)[0])}")

    def save_model_wrapper(self, save_path: str) -> None:
        if self.rank == 0:
            self.logger.info(f"checkpoint saving : {save_path}")
        if self.config.training.use_fsdp:
            self.logger.info(f"dist barrier start {self.rank}")
            dist.barrier()
            self.logger.info(f"dist barrier finish {self.rank}")
            with FSDP.state_dict_type(
                self.model,
                StateDictType.FULL_STATE_DICT,
                FullStateDictConfig(
                    rank0_only=True,
                ),
            ):
                dist.barrier()
                state_dict = self.model.state_dict()
                self.logger.info(f"collected state dict... {self.rank}")

                if self.rank == 0:
                    self.save_model(state_dict, save_path)

            dist.barrier()

        else:
            state_dict = self.model.state_dict()
            if self.rank == 0:
                self.save_model(state_dict, save_path)
            dist.barrier()

    def save_model(self, state_dict: Dict[str, torch.Tensor], save_path: str) -> None:
        torch.save(state_dict, save_path)
        self.logger.info(f"Model has been saved to {save_path}")
