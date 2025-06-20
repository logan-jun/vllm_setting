#!/bin/bash
# 사용법: ./run_vllm.sh 모델이름
# 예시:  ./run_vllm.sh qwen3-0.6b

MODEL_NAME=${1:-qwen3-0.6b}
MODEL_DIR=/models/${MODEL_NAME}
HOST_MODEL_DIR=/home/ubuntu/${MODEL_NAME}

# config.yaml은 현재 디렉터리에 있다고 가정 (필요시 경로 수정)
CONFIG_ON_HOST=./config.yaml
CONFIG_IN_CONTAINER=/app/config.yaml

IMAGE=651706739670.dkr.ecr.ap-northeast-2.amazonaws.com/base_img:qwen3-vllm-openai_2

mkdir -p logs

docker run --gpus all \
  -e VLLM_LOGGING_LEVEL=DEBUG \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --network=host \
  --ipc=host \
  --rm -it \
  -p 8000:8000 \
  -v ${HOST_MODEL_DIR}:${MODEL_DIR} \
  -v ${CONFIG_ON_HOST}:${CONFIG_IN_CONTAINER} \
  -e MODEL_PATH=${MODEL_DIR} \
  ${IMAGE} \
  vllm serve ${MODEL_DIR} --config ${CONFIG_IN_CONTAINER}
  > logs/vllm_debug$(date +%Y%m%d_%H%M%S).log 2>&1
