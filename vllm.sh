#!/usr/bin/env bash

IMAGE_VERSION=latest
IMAGE_NAME=llmconfig-vllm
CONTAINER_NAME=llmconfig-test1
MODEL_DIR=/nas/xq/qwen2.5-7b/Qwen2.5-7B-Instruct
# MODEL_DIR=/data/czh/SFR-Embedding-Mistral
CODE_DIR=/nas/xq/llmconfig-vllm
# DEVICES='"device=0,1,2,3,4,5,6,7"'
DEVICES='"device=6"'

start() {
    echo "start run docker..."
    docker run -itd --name ${CONTAINER_NAME} \
        --log-opt max-size=30m \
        --log-opt max-file=3 \
        --gpus=${DEVICES} \
        --shm-size=16g \
        -p 8000:8000 \
        -p 8001:8001 \
        -p 8002:8002 \
        -e LANG="C.UTF-8" \
        -e LC_ALL="C.UTF-8" \
        -e MODEL_DIR=${MODEL_DIR} \
	    -e WORK_DIR='/workspace/app/llmconfig/python/llmconfig' \
	    -e TASK_TYPE='llm-generate' \
	    -e PARALLEL_SIZE=32 \
        -e MODEL_TYPE="Qwen2" \
        -e PORT=8000 \
        -e KV_CACHE_DTYPE="auto" \
        -e GPU_USAGE=0.9 \
        -e ENABLE_CHUNKED_PREFILL=1 \
	    -e LLM_INFERENCE_URL="http://0.0.0.0:8000" \
        -v ${MODEL_DIR}:${MODEL_DIR} \
        -v ${CODE_DIR}:${CODE_DIR} \
        -v ${CODE_DIR}/llmconfig:/workspace/app/llmconfig1 \
        ${IMAGE_NAME}:${IMAGE_VERSION} /bin/bash
}
#--network host \
$1
