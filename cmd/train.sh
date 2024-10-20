#!/bin/bash

# usage ----------------------------------------------
# bash docker.sh build  # build image
# bash docker.sh shell  # run container as user
# ----------------------------------------------------

train()
{
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    MNCCL_DEBUG=INFO ASTER_ADDR=localhost MASTER_PORT=19999 CUDA_VISIBLE_DEVICES=5 CUDA_LAUNCH_BLOCKING=1 rye run python src/experiment/train/train_fsdp.py --config $CONFIG_PATH
}

docker_train() 
{
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    export NCCL_IB_DISABLE=1
    export NCCL_P2P_DISABLE=1
    CONFIG_PATH=${2:-src/cfg/yaml/template/train_config.yaml}
    NCCL_DEBUG=INFO MASTER_ADDR=localhost MASTER_PORT=19299 CUDA_LAUNCH_BLOCKING=1 python src/experiment/train/train_fsdp.py --config $CONFIG_PATH
}

help()
{
    echo "使用方法: bash train.sh [docker|local|help] [CONFIG_PATH]"
    echo "docker: Dockerコンテナ内でトレーニングを実行します"
    echo "local: ローカル環境でトレーニングを実行します"
    echo "help: このヘルプメッセージを表示します"
    echo "CONFIG_PATH: オプション。使用する設定ファイルのパスを指定します。デフォルトは 'src/cfg/yaml/template/train_config.yaml' です。"
}

if [[ $1 == "docker" ]]; then
    docker_train $2
elif [[ $1 == "local" ]]; then
    train $2
elif [[ $1 == "help" ]]; then
    help
else
    help
fi