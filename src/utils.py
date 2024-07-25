import torch
import wandb

torch_dtype_map: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def wandb_init(config):
    wandb.init(project="xlstm_train", config=config)
