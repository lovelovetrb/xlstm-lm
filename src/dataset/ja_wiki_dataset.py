import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer


class JaWikiDataset:
    def __init__(self, cfg, subset):
        self.cfg = cfg
        self._load_tokenizer()
        self.data = self._load_ja_wiki(subset=subset)

    # NOTE: This method takes time at first, but it's okay.
    def _load_ja_wiki(self, subset) -> list[dict]:
        if self.cfg.dataset.debug:
            dataset_name = "izumi-lab/wikinews-ja-20230728"
        else:
            # https://huggingface.co/datasets/fujiki/wiki40b_ja
            dataset_name = "fujiki/wiki40b_ja"
        row_text_ds = load_dataset(
            dataset_name,
            split=subset,
        )
        tokenized_ds = self._tokenize_dataset(row_text_ds)
        return tokenized_ds

    def _load_tokenizer(self):
        self._tokenizer = AutoTokenizer.from_pretrained(self.cfg.tokenizer.name)
        self.max_seq_length = self.cfg.dataset.max_seq_length
        self.min_seq_length = self.cfg.dataset.min_seq_length

        self._tokenizer.add_special_tokens(
            {"pad_token": "[PAD]", "bos_token": "[BOS]", "eos_token": "[EOS]"}
        )
        self.bos_token_id = self._tokenizer.convert_tokens_to_ids("[BOS]")
        self.eos_token_id = self._tokenizer.convert_tokens_to_ids("[EOS]")
        self.pad_token_id = self._tokenizer.convert_tokens_to_ids("[PAD]")
        assert (
            self.pad_token_id == self.cfg.dataset.pad_token_id
        ), f"PAD token id expect {self.cfg.dataset.pad_token_id}, got {self.pad_token_id}"

    def _tokenize_dataset(self, dataset) -> list[dict]:
        tokenized_ds = []
        for d in tqdm(dataset):
            tokens = self._get_tokens(d["text"].replace("\n", ""))
            if len(tokens["input_ids"][0]) < self.min_seq_length:
                continue

            input_ids, attention_mask = (
                tokens["input_ids"][0],
                tokens["attention_mask"][0],
            )
            data_dic = self._prepare_lm_features_and_labels(input_ids, attention_mask)
            tokenized_ds.append({"text": d["text"], **data_dic})
        return tokenized_ds

    def _get_tokens(self, text):
        return self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_seq_length,
        )

    def _prepare_lm_features_and_labels(
        self, input_ids, attention_mask
    ) -> dict[str, torch.Tensor]:
        def pad_sequence(seq, pad_length, pad_value) -> torch.Tensor:
            return torch.cat([seq, torch.full((pad_length,), pad_value)])

        if len(input_ids) == self.max_seq_length:
            feature_input_ids = torch.cat(
                [torch.tensor([self.bos_token_id]), input_ids[:-1]],
            )
            label_input_ids = input_ids
        elif len(input_ids) == self.max_seq_length - 1:
            feature_input_ids = torch.cat(
                [torch.tensor([self.bos_token_id]), input_ids]
            )
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
                torch.cat([input_ids, torch.tensor([self.eos_token_id])]),
                label_pad_length,
                self.pad_token_id,
            )
        assert (
            feature_input_ids.size(0) == self.max_seq_length
        ), f"Expected length {self.max_seq_length}, got {feature_input_ids.size(0)}"

        return {"feature": feature_input_ids, "label": label_input_ids}
