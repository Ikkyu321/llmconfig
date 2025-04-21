#################### llmconfig installation IMAGE ####################
FROM vllm/vllm-v0.7.3:latest AS llmconfig-vllm

COPY llmconfig /workspace/app/llmconfig

# install llmconfig 
WORKDIR /workspace/app
RUN cd llmconfig/python && python3 -m pip install .


WORKDIR /workspace/app
COPY entrypoint.sh /workspace/app/entrypoint.sh
COPY gpu_count.py /workspace/app/gpu_count.py 
RUN mkdir logs

RUN chmod 755 entrypoint.sh


#################### llmconfig installation IMAGE ####################

ENTRYPOINT ["./entrypoint.sh"]
# ENTRYPOINT [""]
#################### OPENAI API SERVER ####################