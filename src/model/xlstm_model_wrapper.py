import functools

from dacite import Config as DaciteConfig
from dacite import from_dict
from omegaconf import OmegaConf
from torch.distributed.fsdp import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from xlstm import xLSTMLMModel, xLSTMLMModelConfig

from src.utils import torch_dtype_map


class xlstm_model:
    def __init__(self, config, rank) -> None:
        self.config = config
        self.rank = rank
        self.model_cofing = from_dict(
            data_class=xLSTMLMModelConfig,
            data=OmegaConf.to_container(config.model, resolve=True),
            config=DaciteConfig(strict=True),
        )
        self.model = xLSTMLMModel(self.model_cofing)

    def get_model(self):
        self.model = self.model.to(dtype=torch_dtype_map[self.config.training.weight_precision])
        self.model = self.model.to(self.rank)

        if self.config.basic.mode == "train":
            self.model.reset_parameters()
            self.model.train()
        elif self.config.basic.mode == "eval":
            self.model.eval()
        else:
            raise ValueError("Invalid mode")

        if self.config.training.use_fsdp:
            self.model = self._wrap_fsdp(self.model)
        return self.model

    def _wrap_fsdp(self, model):
        return FSDP(
            model,
            sharding_strategy=ShardingStrategy.SHARD_GRAD_OP,
            auto_wrap_policy=self._get_auto_wrap_policy(),
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=self.rank,
            cpu_offload=CPUOffload(offload_params=True),
            param_init_fn=self._param_init_fn,
            limit_all_gathers=True,
            mixed_precision=self._get_mix_precision_policy(),
            use_orig_params=True,
        )

    def _get_auto_wrap_policy(self):
        return functools.partial(
            size_based_auto_wrap_policy,
            # TODO: modelのパラメータ数に応じて調整
            min_num_params=int(5e7),
        )

    def _get_mix_precision_policy(self):
        return MixedPrecision(
            param_dtype=torch_dtype_map[self.config.training.weight_precision],  # weight_precisionがfloat32
            reduce_dtype=torch_dtype_map[self.config.training.amp_precision],  # amp_precisionがbfloat16
            buffer_dtype=torch_dtype_map[self.config.training.amp_precision],  # 通常はreduce_dtypeと同じ
        )

    def _param_init_fn(self, module):
        optim_groups = module._create_weight_decay_optim_groups()
        return (
            {
                "weight_decay": self.config.training.weight_decay,
                "params": optim_groups[0],
            },
            {"weight_decay": 0.0, "params": optim_groups[1]},
        )
