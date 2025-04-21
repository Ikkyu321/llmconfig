FROM python:3.10-slim-buster

ENV TZ=Asia/Shanghai

RUN apt-get update && \
    apt-get install -y --no-install-recommends tzdata coreutils sed && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /home/workspace/app

COPY ./python /home/workspace/app/python
COPY ./entrypoint.sh /home/workspace/app/entrypoint.sh
RUN chmod 755 entrypoint.sh
RUN cd python && pip install -r requirement.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN python setup.py bdist_wheel
ENTRYPOINT [ "bash", "entrypoint.sh" ]