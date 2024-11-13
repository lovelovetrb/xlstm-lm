import pytest
from src.cfg.load_yaml_cfg import load_config

@pytest.fixture
def cfg_paths():
    return {
        # '125m': 'src/cfg/yaml/125m/config.yaml',
        # '350m': 'src/cfg/yaml/350m/config.yaml',
        '1_3b': 'src/cfg/yaml/1.3b/config.yaml'
        # '2_7b': 'src/cfg/yaml/2.7b/config.yaml'
        # 'template': 'src/cfg/yaml/template/config.yaml'
    }


def test_load_1_3b_config(cfg_paths):
    # 1.3Bの設定を読み込む
    config = load_config(cfg_paths['1_3b'])
    # 読み込んだ設定が正しいか確認
    assert config.basic.seed == 42
    assert config.basic.device == 'cuda'
    assert config.model.embedding_dim == 2048
    assert config.training.batch_size == 2
    expect_model_save_dir = f"src/checkpoints/train-{config.basic.project_tag}-{config.dataset.name}"
    assert config.training.model_save_dir == expect_model_save_dir
