from src.cfg.load_yaml_cfg import load_config
from src.experiment.setup.dataloader import get_dataset_generator, setup_dataloader
from src.experiment.setup.tokenizer import setup_tokenizer
from tests.const import cfg_paths
import torch


def test_tokenizer(cfg_paths: dict) -> None:
    # 1.3Bの設定を読み込む
    tokenizer = setup_tokenizer("gpt2")

    config = load_config(cfg_paths["1_3b"])
    generator = get_dataset_generator(config, tokenizer)
    dataloader = setup_dataloader(generator, config, 0, 1, "train")

    sample_num = 50
    for index, batch in enumerate(dataloader):
        feature, label = batch["feature"][0], batch["label"][0]
        feature_text = tokenizer.decode(feature, skip_special_tokens=False)
        # label_text = tokenizer.decode(label, skip_special_tokens=True)
        print(f"Sample {index}")
        print(f"token length: {len(feature)}")
        print(feature_text)
        # print(label_text)
        print("\n\n")
        assert torch.equal(feature[1 : config.dataset.max_seq_length], label[: config.dataset.max_seq_length - 1])
        if index >= sample_num:
            break
