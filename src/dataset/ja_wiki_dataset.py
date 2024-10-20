from datasets import load_dataset


class JaWikiDataset:
    def __init__(self, cfg, subset):
        self.cfg = cfg
        self.data = self._load_ja_wiki(subset=subset)

    def _load_ja_wiki(self, subset) -> list[dict]:
        # https://huggingface.co/datasets/fujiki/wiki40b_ja
        row_text_ds = load_dataset(
            "fujiki/wiki40b_ja",
            split=subset,
            streaming=True,
        )
        return row_text_ds
