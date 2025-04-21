#!/usr/bin/env bash


cd /workspace/app/llmconfig1/python/llmconfig
sed -i 's/"etcdHosts":[^,]*/"etcdHosts":'"\"$ETCD_HOSTS\""'/' meta.json
sed -i 's/"taskType":[^,]*/"taskType":'"\"$TASK_TYPE\""'/' meta.json
#sed -i 's/"llmInferenceUrl":[^,]*/"llmInferenceUrl":'"\"$LLM_INFERENCE_URL\""'/' meta.json
cat meta.json
nohup python3 gradio_webserver.py > gradio.log 2>&1 &
python3 llmconfigcli.py
