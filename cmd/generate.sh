export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=1
CONFIG_PATH=${1:-src/cfg/yaml/template/train_config.yaml}
NCCL_DEBUG=INFO MASTER_ADDR=localhost MASTER_PORT=19299 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0,1,3,4 python src/experiment/test/generate.py --config $CONFIG_PATH