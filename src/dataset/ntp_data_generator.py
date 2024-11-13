import torch
from transformers import AutoTokenizer

from src.cfg.config_type import ExperimentConfig


class NextTokenPredictionDataGenerator:
    def __init__(self, cfg: ExperimentConfig, tokenizer: AutoTokenizer) -> None:
        self.cfg = cfg
        self._tokenizer = tokenizer
        self._load_tokenizer_info()

    def _load_tokenizer_info(self) -> None:
        self.max_seq_length = self.cfg.dataset.max_seq_length
        self.min_seq_length = self.cfg.dataset.min_seq_length

        self.bos_token_id = self._tokenizer.convert_tokens_to_ids("[BOS]")
        self.eos_token_id = self._tokenizer.convert_tokens_to_ids("[EOS]")
        self.pad_token_id = self._tokenizer.convert_tokens_to_ids("[PAD]")

    def tokenize_dataset(self, text: str) -> dict:
        tokens = self._get_tokens(text.replace("\n", ""))
        input_ids = tokens["input_ids"][0]
        return self._prepare_lm_features_and_labels(input_ids)

    def _get_tokens(self, text: str) -> dict:
        return self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_seq_length,
        )

    def _prepare_lm_features_and_labels(self, input_ids: torch.Tensor) -> dict:
        def pad_sequence(seq: torch.Tensor, pad_length: int, pad_value: int) -> torch.Tensor:
            return torch.cat([seq, torch.full((pad_length,), pad_value)])

        if len(input_ids) == self.max_seq_length:
            feature_input_ids = torch.cat(
                [torch.tensor([self.bos_token_id]), input_ids[:-1]],
            )
            label_input_ids = input_ids
        elif len(input_ids) == self.max_seq_length - 1:
            feature_input_ids = torch.cat([torch.tensor([self.bos_token_id]), input_ids])
            label_input_ids = torch.cat([input_ids, torch.tensor([self.eos_token_id])])
        else:
            # NOTE: BOSトークン・EOSトークン分を考慮して、PADトークンを追加する数を計算
            feature_pad_length = self.max_seq_length - len(input_ids) - 2
            feature_input_ids = pad_sequence(
                torch.cat(
                    [
                        torch.tensor([self.bos_token_id]),
                        input_ids,
                        torch.tensor([self.eos_token_id]),
                    ],
                ),
                feature_pad_length,
                self.pad_token_id,
            )
            # NOTE: labelはEOSトークン分のみを考慮して、PADトークンを追加する数を計算
            label_pad_length = self.max_seq_length - len(input_ids) - 1
            label_input_ids = pad_sequence(
                torch.cat(
                    [
                        input_ids,
                        torch.tensor([self.eos_token_id]),
                    ]
                ),
                label_pad_length,
                self.pad_token_id,
            )
        assert (
            feature_input_ids.size(0) == self.max_seq_length
        ), f"Expected length {self.max_seq_length}, got {feature_input_ids.size(0)}"

        return {"feature": feature_input_ids.long(), "label": label_input_ids.long()}
