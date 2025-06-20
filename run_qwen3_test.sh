#!/bin/bash
# 사용법: ./run_vllm.sh 모델이름
# 예시:  ./run_vllm.sh qwen3-0.6b

MODEL_NAME=${1:-qwen3-32b}
MODEL_DIR=/models/${MODEL_NAME}
HOST_MODEL_DIR=/home/ubuntu/${MODEL_NAME}

# 작은 speculative 모델도 마운트
SMALL_MODEL_NAME=qwen3-0.6b
SMALL_MODEL_DIR=/models/${SMALL_MODEL_NAME}
SMALL_HOST_MODEL_DIR=/home/ubuntu/${SMALL_MODEL_NAME}

# config.yaml은 현재 디렉터리에 있다고 가정 (필요시 경로 수정)
CONFIG_ON_HOST=./config_speculative.yaml
CONFIG_IN_CONTAINER=/app/config_speculative.yaml

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
  -v ${SMALL_HOST_MODEL_DIR}:${SMALL_MODEL_DIR} \
  -v ${CONFIG_ON_HOST}:${CONFIG_IN_CONTAINER} \
  -e MODEL_PATH=${MODEL_DIR} \
  ${IMAGE} /bin/bash
