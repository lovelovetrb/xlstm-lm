import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src.dataset.ja_cc_dataset import JaCCDataset
from src.dataset.ja_wiki_dataset import JaWikiDataset
from src.dataset.slim_pajama_dataset import SlimPajamaDataset


class NlpDataset(Dataset):
    def __init__(self, data, cfg):
        self.data = data
        self.cfg = cfg
        self._load_tokenizer()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        tokenized_data = self._tokenize_dataset(self.data[idx]["text"])
        return tokenized_data

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

    def _tokenize_dataset(self, text):
        tokens = self._get_tokens(text.replace("\n", ""))

        input_ids = tokens["input_ids"][0]
        data_dic = self._prepare_lm_features_and_labels(input_ids)
        return data_dic

    def _get_tokens(self, text):
        return self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            add_special_tokens=False,
            max_length=self.max_seq_length,
        )

    def _prepare_lm_features_and_labels(self, input_ids):
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


class NlpDatasetGenerator:
    # TODO: cfgの型を定義
    def __init__(self, cfg):
        self.cfg = cfg
        self.subset = cfg.dataset.subset
        self.datasets = {}
        self._load_data(cfg.dataset.name)

    def _load_data(self, dataset_name: str):
        for subset in self.subset:
            if dataset_name == "slim_pajama":
                # TODO: loggingに変更
                print(
                    f"Loading {subset} dataset from SlimPajama-627B...(this may take a while)"
                )
<<<<<<< Updated upstream
                slimPajamaDataset = SlimPajamaDataset(self.cfg, subset=subset)
                self.datasets[subset] = NlpDataset(slimPajamaDataset.data)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
=======
                slim_pajama_dataset = SlimPajamaDataset(self.cfg, subset=subset)
                self.datasets[subset] = NlpDataset(slim_pajama_dataset.data, self.cfg)
            elif dataset_name == "ja_wiki":
                print(
                    f"Loading {subset} dataset from ja_wiki_40b ...(this may take a while)"
                )
                ja_wiki_dataset = JaWikiDataset(self.cfg, subset=subset)
                self.datasets[subset] = NlpDataset(ja_wiki_dataset.data, self.cfg)
            elif dataset_name == "ja_cc_wiki":
                print(
                    f"Loading {subset} dataset from ja_cc_wiki...(this may take a while)"
                )
                ja_cc_wiki_dataset = JaCCDataset(self.cfg, subset=subset)
                self.datasets[subset] = NlpDataset(ja_cc_wiki_dataset.data, self.cfg)
            else:
                raise ValueError(f"Dataset {dataset_name} not supported")
>>>>>>> Stashed changes

    @property
    def train(self):
        return self.datasets["train"]

    @property
    def valid(self):
        return self.datasets["valid"]

    @property
    def test(self):
        return self.datasets["test"]
