import functools

import torch
from torch.distributed.fsdp import (
    BackwardPrefetch,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from xlstm import xLSTMLMModel

from src.cfg.config_type import ExperimentConfig
from src.utils import get_logger, torch_dtype_map


# ruff: noqa: N801
class xLSTMModelWrapper:
    def __init__(self, config: ExperimentConfig, rank: int, model_weight_path: str | None) -> None:
        self.config = config
        self.logger = get_logger(f"xLSTMModelWrapper: {rank}")
        self.rank = rank
        self.model_weight_path = model_weight_path
        self.model = xLSTMLMModel(self.config.model)

    def get_model(self) -> torch.nn.Module:
        self.load_checkpoint()
        self.model = self.model.to(dtype=torch_dtype_map[self.config.training.weight_precision])
        self.model = self.model.to(self.rank)

        if self.config.basic.mode == "train":
            self.logger.info("Training mode")
            self._train()
        elif self.config.basic.mode == "eval":
            self.logger.info("Evaluation mode")
            self._eval()
        else:
            raise ValueError(self.config.basic.mode)

        if self.config.training.use_fsdp:
            self.model = self._wrap_fsdp(self.model)
        return self.model

    def load_checkpoint(self) -> None:
        if self.model_weight_path is not None:
            self.logger.info(f"Loading checkpoint from {self.model_weight_path}")
            self.model.load_state_dict(torch.load(self.model_weight_path))

    def _train(self) -> None:
        self.model.reset_parameters()
        self.model.train()

    def _eval(self) -> None:
        self.model.eval()

    def _wrap_fsdp(self, model: torch.nn.Module) -> torch.nn.Module:
        self.logger.info("Wrapping model with FSDP")
        return FSDP(
            model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            # sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2,
            auto_wrap_policy=self._get_auto_wrap_policy(),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.rank,
            param_init_fn=self._param_init_fn,
            limit_all_gathers=True,
            mixed_precision=self._get_mix_precision_policy(),
            use_orig_params=True,
        )

    def _get_auto_wrap_policy(self) -> functools.partial:
        return functools.partial(
            size_based_auto_wrap_policy,
            # TODO: modelのパラメータ数に応じて調整
            # https://github.com/lovelovetrb/xlstm-lm/issues/26
            min_num_params=int(1e8),
        )

    def _get_mix_precision_policy(self) -> MixedPrecision:
        return MixedPrecision(
            # weight_precisionがfloat32
            param_dtype=torch_dtype_map[self.config.training.weight_precision],
            # amp_precisionがbfloat16
            reduce_dtype=torch_dtype_map[self.config.training.amp_precision],
            # 通常はreduce_dtypeと同じ
            buffer_dtype=torch_dtype_map[self.config.training.amp_precision],
            keep_low_precision_grads=True,
        )

    def _param_init_fn(self, module: torch.nn.Module) -> tuple:
        # ruff: noqa: SLF001
        optim_groups = module._create_weight_decay_optim_groups()
        return (
            {
                "weight_decay": self.config.training.weight_decay,
                "params": optim_groups[0],
                "foreach": True,
                "fused": True,
            },
            {
                "weight_decay": 0.0,
                "params": optim_groups[1],
                "foreach": True,
                "fused": True,
            },
        )
