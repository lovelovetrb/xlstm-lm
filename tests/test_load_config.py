from src.cfg.load_yaml_cfg import load_config
from tests.const import cfg_paths


def test_load_1_3b_config(cfg_paths: dict) -> None:
    # 1.3Bの設定を読み込む
    config = load_config(cfg_paths["1_3b"])

    # 読み込んだ設定が正しいか確認
    expect_seed = 42
    expect_device = "cuda"
    expect_embedding_dim = 2048
    expect_batch_size = 2
    expect_model_save_dir = f"src/checkpoints/train-{config.basic.project_tag}-{config.dataset.name}"

    assert config.basic.seed == expect_seed
    assert config.basic.device == expect_device
    assert config.model.embedding_dim == expect_embedding_dim
    assert config.training.batch_size == expect_batch_size
    assert config.training.model_save_dir == expect_model_save_dir
