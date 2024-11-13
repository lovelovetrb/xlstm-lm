import torch


def setup_criterion(pad_token_id: int) -> torch.nn.Module:
    return torch.nn.CrossEntropyLoss(ignore_index=pad_token_id)
