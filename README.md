# xLSTM-LM : xLSTMを用いた言語モデルの学習

<a href="https://hub.docker.com/r/continuumio/anaconda3">
    <img src="https://img.shields.io/badge/-Docker-EEE.svg?logo=docker&style=flat">
</a>
    <img src="https://img.shields.io/badge/-Python-EEE.svg?logo=python&style=popout-square">
<a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/-PyTorch-EEE.svg?logo=pytorch&style=popout-square">
</a>
<a href="https://docs.astral.sh/ruff/">
    <img src="https://img.shields.io/badge/-Ruff-EEE.svg?logo=ruff&style=popout-square">
</a>

[![Lint Check](https://github.com/lovelovetrb/xlstm-lm/actions/workflows/lint-python.yml/badge.svg)](https://github.com/lovelovetrb/xlstm-lm/actions/workflows/lint-python.yml)
## 概要
このリポジトリは、xLSTMを用いた言語モデルを学習するコードを実装したものです。  
論文「[xLSTM: Extended Long Short-Term Memory](https://arxiv.org/abs/1911.12488)」に基づいて実装しています。

## ファイル構成
<pre>
.
├── Makefile : Lint・Formatを実行するためのMakefile
├── README.md
├── cmd
│   ├── docker.sh : Dockerコンテナのビルドとコンテナ起動するためのスクリプト
│   └── train.sh : 学習を実行するためのスクリプト
├── docker
│   └── Dockerfile
├── environment.yaml : conda環境の設定ファイル
├── pyproject.toml
├── requirements-dev.lock
├── requirements.lock
├── src
│   ├── cfg
│   │   ├── config_type.py
│   │   ├── load_yaml_cfg.py
│   │   └── yaml : 学習時の設定ファイル
│   │       ├── 1.3b
│   │       │   └── config.yaml
│   │       ├── 125m
│   │       │   └── config.yaml
│   │       ├── 2.7b
│   │       │   └── config.yaml
│   │       ├── 350m
│   │       │   └── config.yaml
│   │       └── template
│   │           └── config.yaml
│   ├── dataset : データセットの読み込みを行うクラス
│   │   ├── ja_cc_dataset.py
│   │   ├── ja_wiki_dataset.py
│   │   ├── nlp_dataset.py
│   │   └── slim_pajama_dataset.py
│   ├── experiment : 実験のためのクラス
│   │   ├── setup
│   │   │   ├── criterion.py
│   │   │   ├── dataset.py
│   │   │   ├── lr_scheduler.py
│   │   │   ├── model.py
│   │   │   └── optimizer.py
│   │   ├── test
│   │   │   └── generate.py
│   │   └── train
│   │       ├── train.py
│   │       └── trainer.py
│   ├── model : モデルの読み込みを行うクラス
│   │   └── xlstm_model_wrapper.py
│   └── utils.py
└── tests
</pre>

## 使用ライブラリ

#### main-dependencies
- [xlstm](https://github.com/NX-AI/xlstm) : NX-AI社が公開しているxLSTMの公式実装
- transformers : tokenizerの読み込みに使用
- datasets : データセットの読み込みに使用
- omegaconf : 設定ファイルの読み込みに使用
- pyTorch : モデルの学習に使用
- wandb : ログの保存に使用

#### dev-dependencies
- ruff : Linter・Formatter
- mypy : 型チェック

## 学習方法
### Dockerを用いたconda環境での学習
1. .envの作成
> wandbのAPIキーを.envに記述してください。
```bash
$ cp .env.sample .env
```
2. Dockerコンテナのビルド
```bash
$ bash cmd/docker.sh build
```
3. Dockerコンテナの起動
```bash
$ bash cmd/docker.sh shell [GPU_ID](ex. 0,1,2,3)
```
4. 学習の実行
```bash
$ bash cmd/train.sh docker [CONFIG_PATH](ex. src/cfg/yaml/1.3b/config.yaml)
```

### ローカル環境での学習
1. 依存関係の解決
- Ryeを用いる場合
```bash
$ rye sync
```
- pipを用いる場合
```bash
$ pip install -r requirements.lock
$ source .venv/bin/activate
```
2. wandbへのログイン
```bash
$ (rye run) wandb login
```
> ※ pipを利用している場合、`(rye run)`を削除してください。

3. 学習の実行
```bash
$ bash cmd/train.sh local [CONFIG_PATH](ex. src/cfg/yaml/1.3b/config.yaml)
```
> ※ pipを利用している場合、cmd/train.shの`rye run python`を`python`に変更してください。

