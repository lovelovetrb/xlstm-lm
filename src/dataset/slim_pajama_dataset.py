from datasets import load_dataset
from transformers import AutoTokenizer


class SlimPajamaDataset:
    def __init__(self, cfg, subset):
        self.cfg = cfg
        self._load_tokenizer()

        self.data = self._load_slim_pajama(subset)

    # NOTE: This method takes time at first, but it's okay.
    def _load_slim_pajama(self, subset) -> list[dict]:
        row_text_ds = load_dataset(
            "cerebras/SlimPajama-627B",
            # TODO: data_filesに対する引数を本番環境用に変更
            # data_files={"train": urls},
            # data_files=f"{subset}/chunk1/example_train_0.jsonl.zst",
            # data_files=f"{subset}/chunk1/*",
            split=subset,
            streaming=True,
        )
        return row_text_ds

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
        print(f"pad_token_id: {self.pad_token_id}")
