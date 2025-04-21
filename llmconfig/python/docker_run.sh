#!/usr/bin/env bash

IMAGE_VERSION=v0.7
IMAGE_NAME=llmconfig
DOCKER_NAME=llmconfig-v0.7.2
CONFIG_FILE=/data/xq/llmconfig

start() {
    # docker start command
    docker run -itd --name ${DOCKER_NAME} \
        --restart=unless-stopped \
        --log-opt max-size=30m \
        --log-opt max-file=3 \
        -p 8001:8001 \
        -p 8002:8002 \
        -e LANG="C.UTF-8" \
        -e LC_ALL="C.UTF-8" \
        -e PARALLEL_SIZE=32 \
        -e MODEL_DIR="/a/b" \
        -e WORK_DIR="/home/workspace/llmconfig/python/llmconfig" \
        -e TASK_TYPE="llm-generate" \
        -e LLM_INFERENCE_URL="http://10.200.99.220:31431/llm/generate" \
        -v /home/xq/llmconfig:/home/workspace/llmconfig \
        ${IMAGE_NAME}:${IMAGE_VERSION}
}

#-e ETCD_HOSTS=${ETCD_HOSTS} \
#-v $CONFIG_FILE:/home/workspace/app/python/config/meta.json \
$1
