import pytest


@pytest.fixture
def cfg_paths() -> dict:
    return {
        "125m": "src/cfg/yaml/125m/config.yaml",
        "350m": "src/cfg/yaml/350m/config.yaml",
        "1_3b": "src/cfg/yaml/1.3b/config.yaml",
        "2_7b": "src/cfg/yaml/2.7b/config.yaml",
        "template": "src/cfg/yaml/template/config.yaml",
    }
