import torch
from src.cfg.load_yaml_cfg import load_config
from src.experiment.setup.dataloader import get_dataset_generator, setup_dataloader
from src.experiment.setup.tokenizer import setup_tokenizer
from tests.const import cfg_paths


def test_dataloader_one_rank(cfg_paths: dict) -> None:
    # 1.3Bの設定を読み込む
    tokenizer = setup_tokenizer("gpt2")

    config = load_config(cfg_paths["1_3b"])
    generator = get_dataset_generator(config, tokenizer)
    dataloader = setup_dataloader(generator, config, 0, 1, "train")

    sample_num = 50
    first_iter = []
    for index, batch in enumerate(dataloader):
        feature, label = batch["feature"][0], batch["label"][0]
        first_iter.append(feature)
        if index >= sample_num:
            break

        # NOTE: データセットが続きから始まるかどうかを確認
    print("Test1: Dataloader is reset.")
    dataloader.dataset.set_start_index(sample_num)
    for index, batch in enumerate(dataloader):
        feature, label = batch["feature"][0], batch["label"][0]
        if index == 10:
            print(f"{index} : {tokenizer.decode(feature, skip_special_tokens=True)}")
            print(f"{index} : {tokenizer.decode(first_iter[index], skip_special_tokens=True)}")
            print("\n\n")
        if torch.equal(feature, first_iter[index]):
            raise ValueError("Dataloader is not shuffled.")
        if index >= sample_num:
            break

    # NOTE: データセットがリセットされるかどうかを確認
    print("Test2: Reset dataloader")
    dataloader.dataset.set_start_index(0)
    for index, batch in enumerate(dataloader):
        feature, label = batch["feature"][0], batch["label"][0]
        if index == 10:
            print(f"{index} : {tokenizer.decode(feature, skip_special_tokens=True)}")
            print(f"{index} : {tokenizer.decode(first_iter[index], skip_special_tokens=True)}")
            print("\n\n")
        if not torch.equal(feature, first_iter[index]):
            raise ValueError("Dataloader is not reset.")
        if index >= sample_num:
            break


def test_dataloader_two_rank(cfg_paths: dict) -> None:
    # 1.3Bの設定を読み込む
    tokenizer = setup_tokenizer("gpt2")

    config = load_config(cfg_paths["1_3b"])
    generator = get_dataset_generator(config, tokenizer)
    dataloader_rank1 = setup_dataloader(generator, config, 0, 2, "train")
    dataloader_rank2 = setup_dataloader(generator, config, 1, 2, "train")

    sample_num = 50
    first_iter_rank_1 = []
    first_iter_rank_2 = []

    for index, (batch1, batch2) in enumerate(zip(dataloader_rank1, dataloader_rank2)):
        feature1, label1 = batch1["feature"][0], batch1["label"][0]
        feature2, label2 = batch2["feature"][0], batch2["label"][0]
        first_iter_rank_1.append(feature1)
        first_iter_rank_2.append(feature2)
        if torch.equal(feature1, feature2):
            raise ValueError("Dataloader is not shuffled.")
        if index >= sample_num:
            break

    # NOTE: データセットが続きから始まるかどうかを確認
    print("Test1: Dataloader is reset.")
    dataloader_rank1.dataset.set_start_index(sample_num)
    dataloader_rank2.dataset.set_start_index(sample_num)
    for index, (batch1, batch2) in enumerate(zip(dataloader_rank1, dataloader_rank2)):
        feature1, label1 = batch1["feature"][0], batch1["label"][0]
        feature2, label2 = batch2["feature"][0], batch2["label"][0]
        if index == 10:
            print(f"{index} : {tokenizer.decode(feature1, skip_special_tokens=True)}")
            print(f"{index} : {tokenizer.decode(first_iter_rank_1[index], skip_special_tokens=True)}")
            print("\n\n")
        if torch.equal(feature1, first_iter_rank_1[index]):
            raise ValueError("Dataloader is not shuffled.")
        if torch.equal(feature2, first_iter_rank_2[index]):
            raise ValueError("Dataloader is not shuffled.")
        if torch.equal(feature1, feature2):
            raise ValueError("Dataloader is not shuffled.")
        if index >= sample_num:
            break
