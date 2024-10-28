#!/bin/bash

# usage ----------------------------------------------
# bash docker.sh build  # build image
# bash docker.sh shell  # run container as user
# ----------------------------------------------------

DOCKERFILE_NAME="Dockerfile"
IMAGE_NAME="xlstm-train"

build()
{
    export DOCKER_BUILDKIT=1 
    docker build . -f docker/$DOCKERFILE_NAME -t $IMAGE_NAME
}

shell() 
{
    GPU_ID=${1:-all}
    echo "USE GPU_ID: $GPU_ID"
    docker run --shm-size=10g --gpus \""device=$GPU_ID"\" -it -v $(pwd):/app --env-file ./.env $IMAGE_NAME:latest
}

help(){
    echo "使用方法: bash docker.sh [build|shell|help] [GPU_ID]"
    echo "GPU_ID: オプション。使用するGPU IDを指定します。デフォルトは'all'です。"
}


if [[ $1 == "build" ]]; then
    build
elif [[ $1 == "shell" ]]; then
    shell $2
elif [[ $1 == "help" ]]; then
    help
else
    help
fi